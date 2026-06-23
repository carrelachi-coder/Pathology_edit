#!/usr/bin/env python
"""Generate I0 with Cross V1 ControlNet, then run RF-Solver CIA.

This is a two-stage experiment wrapper:

1. Use the metadata reference/target pair to generate an I0 image with the
   existing Cross V1 ControlNet checkpoint.
2. Use I0 as the structure image for RF-Solver inversion, then denoise from
   the I0 inversion noise with ControlNet while injecting reference K/V.

The wrapper does not modify RF-Solver-Edit. It saves I0 first, releases the
ControlNet bundle, and then launches the RF-Solver/ControlNet runner in a
subprocess.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PROMPT = "H&E stained cancer histopathology at 40x magnification"
NUCLEI_STAIN_LABEL_OFFSET = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage gate: Cross V1 ControlNet generates I0 from a metadata "
            "ref/target pair, then RF-Solver inversion + ControlNet denoise "
            "injects reference K/V into I0."
        )
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="metadata_cross_val.json / metadata_cross_train.json.",
    )
    parser.add_argument("--metadata-index", type=int, default=0)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument(
        "--pretrained-model-name-or-path",
        default=os.environ.get("MODEL_DIR") or os.environ.get("FLUX_DIFFUSERS_ROOT"),
        required=False,
        help="Diffusers FLUX root for the ControlNet pipeline.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CONTROLNET_CHECKPOINT")
        or os.environ.get("CONTROLNET_CKPT"),
        required=False,
        help="Cross V1 ControlNet checkpoint directory.",
    )
    parser.add_argument(
        "--uni-checkpoint-path",
        default=os.environ.get("UNI_CHECKPOINT") or os.environ.get("UNI_CKPT"),
        required=False,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--controlnet-num-inference-steps", type=int, default=28)
    parser.add_argument("--controlnet-guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument(
        "--i0-source",
        choices=("controlnet", "target", "explicit"),
        default="controlnet",
        help=(
            "Source for the Stage 2 I0 image. controlnet generates I0 with "
            "Cross V1; target uses metadata target_image directly; explicit "
            "uses --i0-image."
        ),
    )
    parser.add_argument(
        "--i0-image",
        type=Path,
        default=None,
        help="Explicit I0 image used when --i0-source explicit.",
    )
    parser.add_argument("--regional-ip-soft-bias", type=float, default=None)
    parser.add_argument("--source-latent-init-strength", type=float, default=0.0)
    parser.add_argument("--mask-chord-scale", type=float, default=0.0)
    parser.add_argument("--mask-chord-use-gate", action="store_true")
    parser.add_argument("--mask-chord-gate-dilate-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-feather-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-outside-scale", type=float, default=0.0)
    parser.add_argument(
        "--controlnet-color-match",
        choices=("none", "lab", "macenko", "hed", "hd"),
        default="none",
        help=(
            "Optional stain/color postprocess on I0 before RF-Solver. Use "
            "macenko for H&E stain transfer; hed/hd are accepted aliases for "
            "the same reference-stain matching path."
        ),
    )
    parser.add_argument(
        "--controlnet-color-match-scope",
        choices=("region", "global"),
        default="region",
        help=(
            "For Macenko stain matching, match by tissue+nuclei composite masks "
            "or over all non-background tissue."
        ),
    )
    parser.add_argument("--controlnet-color-match-strength", type=float, default=1.0)
    parser.add_argument("--controlnet-color-match-background-label", type=int, default=0)
    parser.add_argument(
        "--controlnet-color-match-fallback",
        choices=("pooled", "skip"),
        default="pooled",
        help="Fallback for regional stain labels missing from the reference mask.",
    )
    parser.add_argument("--controlnet-macenko-io", type=float, default=240.0)
    parser.add_argument("--controlnet-macenko-beta", type=float, default=0.15)
    parser.add_argument("--controlnet-macenko-alpha", type=float, default=1.0)

    parser.add_argument(
        "--prompt-source",
        choices=("metadata", "dataset", "empty"),
        default="metadata",
        help="Prompt for ControlNet I0 and default RF source/reference prompts.",
    )
    parser.add_argument("--prompt", default=None, help="Override prompt for both stages.")
    parser.add_argument("--rf-source-prompt", default=None)
    parser.add_argument("--rf-target-prompt", default=None)
    parser.add_argument("--rf-reference-prompt", default=None)

    parser.add_argument(
        "--rf-stage-mode",
        choices=("controlnet", "pure_flux"),
        default="controlnet",
        help=(
            "Stage 2 runner. controlnet does pure FLUX inversion of I0 and "
            "ControlNet denoise with late K/V injection. pure_flux keeps the "
            "older RF-Solver-only CIA path."
        ),
    )
    parser.add_argument(
        "--rf-script",
        type=Path,
        default=None,
        help=(
            "Optional Stage 2 runner path. Defaults to "
            "rf_solver_flux_controlnet_reconstruct.py for --rf-stage-mode "
            "controlnet, otherwise rf_solver_flux_reconstruct.py."
        ),
    )
    parser.add_argument("--rf-solver-root", type=Path, default=_env_path("RF_SOLVER_ROOT"))
    parser.add_argument(
        "--rf-output-subdir",
        default="rf_solver_cross",
        help="Subdirectory under --output-dir for RF-Solver artifacts.",
    )
    parser.add_argument("--rf-name", default="flux-dev")
    parser.add_argument("--rf-device", default=None, help="Defaults to --device.")
    parser.add_argument("--rf-offload", action="store_true")
    parser.add_argument("--rf-num-inference-steps", type=int, default=25)
    parser.add_argument("--rf-guidance", type=float, default=1.0)
    parser.add_argument("--rf-inversion-guidance", type=float, default=1.0)
    parser.add_argument("--rf-baseline-guidance", type=float, default=1.0)
    parser.add_argument(
        "--rf-save-rf-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In ControlNet Stage 2, also save the pure RF baseline from I0 zT.",
    )
    parser.add_argument("--rf-controlnet-guidance-scale", type=float, default=1.0)
    parser.add_argument("--rf-controlnet-conditioning-scale", type=float, default=0.2)
    parser.add_argument(
        "--rf-controlnet-start-step",
        type=int,
        default=18,
        help=(
            "In ControlNet Stage 2, disable ControlNet residuals before this "
            "denoise step. Late/weak is usually more stable for pathology."
        ),
    )
    parser.add_argument(
        "--rf-controlnet-reference-source",
        choices=("self", "metadata"),
        default="self",
        help=(
            "ControlNet reference during I0->CIA denoise. self anchors to I0; "
            "metadata reuses the metadata reference image and is more confounded "
            "with texture transfer."
        ),
    )
    parser.add_argument(
        "--rf-ip-scale",
        type=float,
        default=0.0,
        help="Cross V1 IP scale inside Stage 2 ControlNet denoise.",
    )
    parser.add_argument("--rf-regional-ip-soft-bias", type=float, default=None)
    parser.add_argument(
        "--rf-with-second-order",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rf-cross-image-mode",
        choices=("v-only", "kv", "both"),
        default="kv",
    )
    parser.add_argument("--rf-cross-image-strength", type=float, default=1.0)
    parser.add_argument("--rf-inject-steps", type=int, default=0)
    parser.add_argument(
        "--rf-inject-after-t",
        type=float,
        default=None,
        help=(
            "Optional Stage 2 K/V injection timestep threshold. When omitted "
            "for --rf-stage-mode controlnet, the downstream runner derives it "
            "from --rf-kv-inject-start-step, matching direct "
            "rf_solver_flux_controlnet_reconstruct.py runs."
        ),
    )
    parser.add_argument(
        "--rf-kv-inject-start-step",
        type=int,
        default=18,
        help=(
            "In ControlNet Stage 2, first denoise step where reference K/V "
            "injection is enabled."
        ),
    )
    parser.add_argument("--rf-cross-after-layer", type=int, default=20)
    parser.add_argument(
        "--rf-regional-mode",
        "--regional-mode",
        dest="rf_regional_mode",
        choices=("none", "tissue", "nuclei", "tissue_nuclei"),
        default="none",
        help=(
            "RF-Solver STEP 4 regional mode. --regional-mode is accepted as a "
            "convenience alias for this wrapper."
        ),
    )
    parser.add_argument(
        "--rf-kv-protect-target-nuclei",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward --kv-protect-target-nuclei to the ControlNet Stage 2 CIA "
            "runner. Only applies with --rf-stage-mode controlnet."
        ),
    )
    parser.add_argument(
        "--rf-kv-target-nuclei-inject-scale",
        type=float,
        default=0.0,
        help="Forward --kv-target-nuclei-inject-scale to the Stage 2 runner.",
    )
    parser.add_argument(
        "--rf-kv-block-ref-nuclei-to-target-non-nuclei",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward --kv-block-ref-nuclei-to-target-non-nuclei to the "
            "ControlNet Stage 2 CIA runner."
        ),
    )
    parser.add_argument("--rf-kv-nuclei-occupancy-dilate-px", type=int, default=4)
    parser.add_argument("--rf-kv-nuclei-occupancy-min-pixels", type=int, default=3)
    parser.add_argument("--rf-kv-nuclei-occupancy-min-fraction", type=float, default=0.01)
    parser.add_argument("--rf-save-feature-debug", action="store_true")
    parser.add_argument("--rf-debug-features", action="store_true")
    parser.add_argument("--rf-fail-on-threshold", action="store_true")
    parser.add_argument("--rf-allow-text-encoder-download", action="store_true")
    parser.add_argument(
        "--rf-kv-reference-preprocess",
        choices=("none", "inpaint_ref_nuclei"),
        default="none",
        help=(
            "Optional preprocessing for the K/V appearance reference. "
            "inpaint_ref_nuclei removes reference nuclei before RF inversion so "
            "CIA transfers tissue texture without copying reference nuclei."
        ),
    )
    parser.add_argument(
        "--rf-kv-reference-nuclei-mask",
        type=Path,
        default=None,
        help=(
            "Nuclei mask used by --rf-kv-reference-preprocess. Defaults to "
            "metadata reference_nuclei_mask. Also forwarded as the Stage 2 "
            "K/V reference nuclei mask when provided."
        ),
    )
    parser.add_argument(
        "--rf-kv-reference-tissue-mask",
        type=Path,
        default=None,
        help=(
            "Optional tissue mask for the Stage 2 K/V reference. Defaults to "
            "the selected metadata reference tissue mask."
        ),
    )
    parser.add_argument("--rf-kv-inpaint-radius", type=float, default=5.0)
    parser.add_argument("--rf-kv-inpaint-dilate", type=int, default=2)
    parser.add_argument(
        "--rf-kv-inpaint-method",
        choices=("telea", "ns"),
        default="telea",
    )
    parser.add_argument(
        "--rf-cia-sweep-preset",
        choices=("none", "texture_probe", "aggressive"),
        default="none",
        help=(
            "Run multiple Stage 2 ControlNet-CIA settings from the same I0. "
            "texture_probe starts with moderate stronger injection; aggressive "
            "adds one high-risk probe."
        ),
    )
    parser.add_argument(
        "--rf-sweep-configs",
        default=None,
        help=(
            "Optional comma-separated Stage 2 configs. Each item is "
            "colon-separated key=value fields, e.g. "
            "'tag=mid:s=0.5:l=14:ks=10:t=0.7:cn=19:c=0.15'. "
            "Keys: tag,s,l,ks,t,cn,c,ip,mode."
        ),
    )
    parser.add_argument(
        "--rf-sweep-stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the sweep on the first failed Stage 2 run.",
    )

    parser.add_argument(
        "--rf-reference-image",
        type=Path,
        default=None,
        help="Override RF appearance reference image. Default uses metadata reference_image.",
    )
    parser.add_argument(
        "--rf-reference-sample-id",
        default=None,
        help="Metadata sample_id to use as RF appearance reference.",
    )
    parser.add_argument("--rf-reference-metadata-index", type=int, default=None)
    parser.add_argument(
        "--rf-reference-record-image-field",
        choices=("target_image", "reference_image"),
        default="target_image",
    )
    parser.add_argument("--rf-auto-reference-by-texture", action="store_true")
    parser.add_argument("--rf-auto-reference-max-candidates", type=int, default=300)
    parser.add_argument("--rf-auto-reference-rank", type=int, default=0)
    parser.add_argument(
        "--rf-auto-reference-same-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rf-auto-reference-different-case",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--flux-diffusers-root", type=Path, default=_env_path("FLUX_DIFFUSERS_ROOT"))
    parser.add_argument("--t5-model-path", type=Path, default=_env_path("T5_MODEL_PATH"))
    parser.add_argument("--t5-tokenizer-path", type=Path, default=_env_path("T5_TOKENIZER_PATH"))
    parser.add_argument("--clip-model-path", type=Path, default=_env_path("CLIP_MODEL_PATH"))
    parser.add_argument("--clip-tokenizer-path", type=Path, default=_env_path("CLIP_TOKENIZER_PATH"))

    parser.add_argument(
        "--skip-rf-solver",
        action="store_true",
        help="Only generate I0 and summary; useful for checking the ControlNet stage.",
    )
    parser.add_argument(
        "--dry-run-rf-command",
        action="store_true",
        help="Generate I0 and print/write the RF command without executing it.",
    )
    parser.add_argument(
        "--post-controlnet-reanchor",
        action="store_true",
        help=(
            "After RF-Solver cross-image output, run a low-strength Cross V1 "
            "ControlNet img2img-style pass from the RF output latent to pull "
            "natural-image artifacts back toward the pathology manifold."
        ),
    )
    parser.add_argument(
        "--post-controlnet-source-image",
        type=Path,
        default=None,
        help="Optional explicit image to re-anchor. Defaults to the RF cross output.",
    )
    parser.add_argument(
        "--post-controlnet-source",
        choices=("auto", "cross_kv", "cross_v_only", "baseline"),
        default="auto",
        help="Which RF artifact to use as the re-anchor source when no explicit image is supplied.",
    )
    parser.add_argument("--post-controlnet-output-name", default="controlnet_reanchored.png")
    parser.add_argument("--post-controlnet-num-inference-steps", type=int, default=18)
    parser.add_argument("--post-controlnet-guidance-scale", type=float, default=3.0)
    parser.add_argument("--post-controlnet-conditioning-scale", type=float, default=0.6)
    parser.add_argument(
        "--post-controlnet-strength",
        type=float,
        default=0.35,
        help=(
            "Img2img source latent strength in [0,1]. Lower preserves the RF "
            "texture more; higher lets ControlNet repair harder artifacts."
        ),
    )
    parser.add_argument(
        "--post-controlnet-ip-scale",
        type=float,
        default=0.5,
        help="Reference IP scale for the post ControlNet re-anchor pass.",
    )
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument(
        "--save-cia-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save I0-vs-CIA diff, metrics, and panel after RF-Solver CIA postprocess.",
    )
    return parser


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    started_at = time.perf_counter()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    record, metadata_index = select_metadata_record(args.metadata, args.sample_id, args.metadata_index)
    prompt = resolve_prompt(record, prompt_override=args.prompt, prompt_source=args.prompt_source)

    sample_id = str(record.get("sample_id") or Path(record["target_image"]).stem)
    ref_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
    controlnet_dir = output_dir / "controlnet_i0"
    controlnet_dir.mkdir(parents=True, exist_ok=True)

    target_gt_path = copy_image(record["target_image"], controlnet_dir / "target_gt.png")
    reference_path = copy_image(record["reference_image"], controlnet_dir / "controlnet_reference.png")
    copy_image(record["target_tissue_mask"], controlnet_dir / "target_tissue_mask.png")
    copy_image(record["target_nuclei_mask"], controlnet_dir / "target_nuclei_mask.png")
    copy_image(record["reference_tissue_mask"], controlnet_dir / "reference_tissue_mask.png")
    copy_image(record["reference_nuclei_mask"], controlnet_dir / "reference_nuclei_mask.png")

    if args.i0_source == "controlnet":
        print(
            "Stage 1: generating I0 with Cross V1 ControlNet "
            f"sample={sample_id} ref={ref_id} seed={args.seed}"
        )
        i0_raw, controlnet_summary = generate_controlnet_i0(
            args=args,
            record=record,
            prompt=prompt,
        )
    else:
        source_path = Path(record["target_image"]) if args.i0_source == "target" else args.i0_image
        if source_path is None:
            raise ValueError("--i0-source explicit requires --i0-image.")
        print(
            "Stage 1: bypassing ControlNet I0 generation; "
            f"using {args.i0_source} image as I0: {source_path}"
        )
        i0_raw = load_rgb(source_path)
        controlnet_summary = {
            "bypassed": True,
            "i0_source": args.i0_source,
            "i0_image": str(source_path),
            "checkpoint": None,
            "seed": int(args.seed),
        }
    i0_raw_path = controlnet_dir / "controlnet_i0_raw.png"
    i0_raw.save(i0_raw_path)
    if args.i0_source == "controlnet":
        i0, stain_match_summary = apply_controlnet_stain_match(
            args=args,
            i0=i0_raw,
            record=record,
            output_dir=controlnet_dir,
        )
    else:
        i0 = i0_raw.convert("RGB")
        stain_match_summary = {
            "mode": "none",
            "applied": False,
            "reason": f"i0_source_{args.i0_source}_bypasses_controlnet_postprocess",
            "input": "controlnet_i0_raw.png",
            "output": "controlnet_i0.png",
        }
    i0_path = controlnet_dir / "controlnet_i0.png"
    i0.save(i0_path)
    controlnet_summary["stain_match"] = stain_match_summary

    save_controlnet_panel(
        reference=Image.open(reference_path).convert("RGB"),
        target=Image.open(target_gt_path).convert("RGB"),
        i0_raw=i0_raw,
        i0=i0,
        output_path=controlnet_dir / "controlnet_i0_panel.png",
        thumbnail_size=args.thumbnail_size,
        title=f"{sample_id} | ref={ref_id}",
    )

    rf_output_dir = output_dir / args.rf_output_subdir
    kv_reference_preprocess: dict[str, Any] = {"mode": args.rf_kv_reference_preprocess, "applied": False}
    if args.rf_kv_reference_preprocess != "none":
        kv_reference_preprocess = prepare_rf_kv_reference(
            args=args,
            record=record,
            output_dir=output_dir / "rf_reference_preprocess",
        )
    rf_command = build_rf_command(
        args=args,
        record=record,
        i0_path=i0_path,
        rf_output_dir=rf_output_dir,
        prompt=prompt,
    )

    rf_result: dict[str, Any] = {
        "skipped": bool(args.skip_rf_solver),
        "dry_run": bool(args.dry_run_rf_command),
        "returncode": None,
        "kv_reference_preprocess": kv_reference_preprocess,
    }
    if args.skip_rf_solver:
        print("Stage 2: skipped RF-Solver by request.")
    elif args.dry_run_rf_command:
        print("Stage 2 RF command:")
        print(" ".join(shell_quote(part) for part in rf_command))
    else:
        if args.rf_stage_mode == "controlnet":
            print(
                "Stage 2: running RF inversion + in-loop ControlNet CIA "
                "from generated I0..."
            )
        else:
            print("Stage 2: running RF-Solver-only CIA from generated I0...")
        release_cuda_memory()
        completed = subprocess.run(rf_command, check=False)
        rf_result["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            print(f"RF-Solver stage failed with return code {completed.returncode}.")
        if int(rf_result.get("returncode") or 0) != 0:
            write_summary(
                output_dir=output_dir,
                args=args,
                record=record,
                metadata_index=metadata_index,
                prompt=prompt,
                controlnet_summary=controlnet_summary,
                i0_raw_path=i0_raw_path,
                i0_path=i0_path,
                rf_command=rf_command,
                rf_output_dir=rf_output_dir,
                rf_result=rf_result,
                started_at=started_at,
            )
            return int(rf_result["returncode"])

    post_reanchor_summary: dict[str, Any] | None = None
    post_reanchor_path: Path | None = None
    cia_diagnostics: dict[str, Any] | None = None
    if (
        args.save_cia_diagnostics
        and not args.skip_rf_solver
        and not args.dry_run_rf_command
        and rf_output_dir.exists()
    ):
        cia_diagnostics = save_cia_postprocess_diagnostics(
            output_dir=output_dir / "cia_postprocess_diagnostics",
            target_path=Path(record["target_image"]),
            reference_path=Path(record["reference_image"]),
            i0_path=i0_path,
            rf_output_dir=rf_output_dir,
            thumbnail_size=args.thumbnail_size,
        )
        rf_result["cia_postprocess_diagnostics"] = cia_diagnostics

    if args.post_controlnet_reanchor and not args.dry_run_rf_command:
        print("Stage 3: running low-strength ControlNet pathology re-anchor...")
        post_reanchor_summary = run_post_controlnet_reanchor(
            args=args,
            record=record,
            prompt=prompt,
            rf_output_dir=rf_output_dir,
            output_dir=output_dir / "post_controlnet_reanchor",
        )
        post_reanchor_image = post_reanchor_summary.get("artifacts", {}).get("reanchored")
        if post_reanchor_image:
            post_reanchor_path = Path(str(post_reanchor_image))
        rf_result["post_controlnet_reanchor"] = post_reanchor_summary

    final_panel = save_final_panel(
        output_dir=output_dir,
        target_path=Path(record["target_image"]),
        reference_path=Path(record["reference_image"]),
        i0_path=i0_path,
        rf_output_dir=rf_output_dir,
        post_reanchor_path=post_reanchor_path,
        thumbnail_size=args.thumbnail_size,
    )
    if final_panel is not None:
        rf_result["final_panel"] = str(final_panel)

    summary_path = write_summary(
        output_dir=output_dir,
        args=args,
        record=record,
        metadata_index=metadata_index,
        prompt=prompt,
        controlnet_summary=controlnet_summary,
        i0_raw_path=i0_raw_path,
        i0_path=i0_path,
        rf_command=rf_command,
        rf_output_dir=rf_output_dir,
        rf_result=rf_result,
        started_at=started_at,
    )
    print(f"Saved two-stage summary to {summary_path}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.rf_script is None:
        default_script = (
            "rf_solver_flux_controlnet_reconstruct.py"
            if args.rf_stage_mode == "controlnet"
            else "rf_solver_flux_reconstruct.py"
        )
        args.rf_script = Path(__file__).resolve().with_name(default_script)
    needs_controlnet_inputs = args.i0_source == "controlnet" or args.rf_stage_mode == "controlnet" or args.post_controlnet_reanchor
    missing = []
    if needs_controlnet_inputs and not args.pretrained_model_name_or_path:
        missing.append("--pretrained-model-name-or-path or MODEL_DIR/FLUX_DIFFUSERS_ROOT")
    if needs_controlnet_inputs and not args.checkpoint:
        missing.append("--checkpoint or CONTROLNET_CHECKPOINT/CONTROLNET_CKPT")
    if needs_controlnet_inputs and not args.uni_checkpoint_path:
        missing.append("--uni-checkpoint-path or UNI_CHECKPOINT/UNI_CKPT")
    if missing:
        raise ValueError("Missing required ControlNet inputs: " + ", ".join(missing))
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {args.metadata}")
    if not args.rf_script.exists():
        raise FileNotFoundError(f"RF-Solver runner does not exist: {args.rf_script}")
    if args.controlnet_num_inference_steps <= 0:
        raise ValueError("--controlnet-num-inference-steps must be positive.")
    if args.i0_source == "explicit":
        if args.i0_image is None:
            raise ValueError("--i0-source explicit requires --i0-image.")
        if not args.i0_image.exists():
            raise FileNotFoundError(f"--i0-image does not exist: {args.i0_image}")
    if args.rf_num_inference_steps <= 0:
        raise ValueError("--rf-num-inference-steps must be positive.")
    if not (0.0 <= args.rf_cross_image_strength <= 1.0):
        raise ValueError("--rf-cross-image-strength must be in [0, 1].")
    if args.rf_controlnet_start_step < 0:
        raise ValueError("--rf-controlnet-start-step must be >= 0.")
    if args.rf_kv_inject_start_step < 0:
        raise ValueError("--rf-kv-inject-start-step must be >= 0.")
    if args.rf_cross_after_layer < 0:
        raise ValueError("--rf-cross-after-layer must be >= 0.")
    if args.rf_stage_mode == "controlnet" and args.rf_cross_image_mode == "both":
        raise ValueError(
            "--rf-cross-image-mode both is supported only with --rf-stage-mode pure_flux. "
            "For the in-loop ControlNet CIA runner, run v-only and kv as two separate jobs."
        )
    if args.rf_stage_mode != "controlnet" and (
        args.rf_kv_protect_target_nuclei
        or args.rf_kv_block_ref_nuclei_to_target_non_nuclei
    ):
        raise ValueError(
            "RF K/V nuclei protection is implemented only for "
            "--rf-stage-mode controlnet."
        )
    if not (0.0 <= args.rf_kv_target_nuclei_inject_scale <= 1.0):
        raise ValueError("--rf-kv-target-nuclei-inject-scale must be in [0, 1].")
    if args.rf_kv_nuclei_occupancy_dilate_px < 0:
        raise ValueError("--rf-kv-nuclei-occupancy-dilate-px must be >= 0.")
    if args.rf_kv_nuclei_occupancy_min_pixels < 0:
        raise ValueError("--rf-kv-nuclei-occupancy-min-pixels must be >= 0.")
    if not (0.0 <= args.rf_kv_nuclei_occupancy_min_fraction <= 1.0):
        raise ValueError("--rf-kv-nuclei-occupancy-min-fraction must be in [0, 1].")
    if args.rf_kv_inpaint_radius <= 0:
        raise ValueError("--rf-kv-inpaint-radius must be positive.")
    if args.rf_kv_inpaint_dilate < 0:
        raise ValueError("--rf-kv-inpaint-dilate must be >= 0.")
    if args.rf_cia_sweep_preset != "none" or args.rf_sweep_configs:
        raise ValueError(
            "RF CIA sweep is disabled in this branch while testing tissue-only "
            "reference preprocessing. Run one configuration per command."
        )
    if not (0.0 <= args.post_controlnet_strength <= 1.0):
        raise ValueError("--post-controlnet-strength must be in [0, 1].")
    if args.post_controlnet_num_inference_steps <= 0:
        raise ValueError("--post-controlnet-num-inference-steps must be positive.")


def read_metadata_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs") or payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Metadata object must contain a pairs/records list: {path}")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported metadata payload type: {type(payload)!r}")


def select_metadata_record(
    metadata_path: Path,
    sample_id: str | None,
    metadata_index: int,
) -> tuple[dict[str, Any], int]:
    records = read_metadata_records(metadata_path)
    if not records:
        raise ValueError(f"Metadata file is empty: {metadata_path}")
    if sample_id is not None:
        for index, record in enumerate(records):
            if str(record.get("sample_id")) == sample_id:
                return record, index
        raise ValueError(f"sample_id {sample_id!r} not found in {metadata_path}")
    if metadata_index < 0 or metadata_index >= len(records):
        raise IndexError(
            f"--metadata-index {metadata_index} out of range for {len(records)} records"
        )
    return records[metadata_index], metadata_index


def resolve_prompt(
    record: dict[str, Any],
    *,
    prompt_override: str | None,
    prompt_source: str,
) -> str:
    if prompt_override is not None:
        return prompt_override
    if prompt_source == "empty":
        return ""
    if prompt_source == "metadata":
        return str(record.get("prompt") or DEFAULT_PROMPT)
    if prompt_source == "dataset":
        from controlnet_train.data.common import default_prompt_for_dataset

        dataset = record.get("dataset")
        if dataset:
            return default_prompt_for_dataset(str(dataset))
    return DEFAULT_PROMPT


def copy_image(source: str | Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        ImageOps.exif_transpose(image).save(destination)
    return destination


def generate_controlnet_i0(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    prompt: str,
) -> tuple[Image.Image, dict[str, Any]]:
    import torch

    from controlnet_train.data.common import (
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v1 import (
        load_cross_v1_bundle,
        run_cross_v1_bundle,
        set_ip_soft_bias,
    )

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=str(args.pretrained_model_name_or_path),
        checkpoint_path=str(args.checkpoint),
        uni_checkpoint_path=str(args.uni_checkpoint_path),
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.controlnet_num_inference_steps,
        guidance_scale=args.controlnet_guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_scale,
    )
    soft_bias_override = None
    if args.regional_ip_soft_bias is not None:
        soft_bias_override = set_ip_soft_bias(
            bundle.flux_pipeline.transformer,
            args.regional_ip_soft_bias,
        )
        print(
            "regional_ip_soft_bias override "
            f"requested={args.regional_ip_soft_bias:g} "
            f"applied={soft_bias_override['applied']} "
            f"params={soft_bias_override['parameter_count']}"
        )

    reference_image = load_image_tensor(record["reference_image"])
    reference_tissue_mask = load_tissue_mask(record["reference_tissue_mask"])
    reference_nuclei_mask = load_nuclei_mask(record["reference_nuclei_mask"])
    target_tissue_mask = load_tissue_mask(record["target_tissue_mask"])
    target_nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"])

    with torch.no_grad():
        i0 = run_cross_v1_bundle(
            bundle,
            reference_image=reference_image,
            reference_tissue_mask=reference_tissue_mask,
            reference_nuclei_mask=reference_nuclei_mask,
            target_tissue_mask=target_tissue_mask,
            target_nuclei_mask=target_nuclei_mask,
            prompt=prompt,
            source_latent_init_strength=args.source_latent_init_strength,
            mask_chord_scale=args.mask_chord_scale,
            mask_chord_use_gate=args.mask_chord_use_gate,
            mask_chord_gate_dilate_radius=args.mask_chord_gate_dilate_radius,
            mask_chord_gate_feather_radius=args.mask_chord_gate_feather_radius,
            mask_chord_gate_outside_scale=args.mask_chord_gate_outside_scale,
            seed=args.seed,
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "uni_checkpoint_path": str(args.uni_checkpoint_path),
        "num_inference_steps": int(args.controlnet_num_inference_steps),
        "guidance_scale": float(args.controlnet_guidance_scale),
        "controlnet_conditioning_scale": float(args.controlnet_conditioning_scale),
        "ip_scale": float(args.ip_scale),
        "regional_ip_soft_bias": (
            float(args.regional_ip_soft_bias)
            if args.regional_ip_soft_bias is not None
            else None
        ),
        "regional_ip_soft_bias_override": soft_bias_override,
        "source_latent_init_strength": float(args.source_latent_init_strength),
        "mask_chord_scale": float(args.mask_chord_scale),
        "mask_chord_use_gate": bool(args.mask_chord_use_gate),
        "mask_chord_gate_dilate_radius": int(args.mask_chord_gate_dilate_radius),
        "mask_chord_gate_feather_radius": int(args.mask_chord_gate_feather_radius),
        "mask_chord_gate_outside_scale": float(args.mask_chord_gate_outside_scale),
        "seed": int(args.seed),
        "color_match": args.controlnet_color_match,
    }

    # Drop the largest references before the RF subprocess starts.
    del bundle
    release_cuda_memory()
    return i0.convert("RGB"), summary


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def run_post_controlnet_reanchor(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    prompt: str,
    rf_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    from controlnet_train.data.common import (
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v1 import (
        load_cross_v1_bundle,
        run_cross_v1_bundle,
        set_ip_soft_bias,
    )

    source_path = resolve_post_controlnet_source(
        explicit=args.post_controlnet_source_image,
        rf_output_dir=rf_output_dir,
        source=args.post_controlnet_source,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_copy = copy_image(source_path, output_dir / "reanchor_source.png")

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=str(args.pretrained_model_name_or_path),
        checkpoint_path=str(args.checkpoint),
        uni_checkpoint_path=str(args.uni_checkpoint_path),
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.post_controlnet_num_inference_steps,
        guidance_scale=args.post_controlnet_guidance_scale,
        controlnet_conditioning_scale=args.post_controlnet_conditioning_scale,
        ip_adapter_scale=args.post_controlnet_ip_scale,
    )
    soft_bias_override = None
    if args.regional_ip_soft_bias is not None:
        soft_bias_override = set_ip_soft_bias(
            bundle.flux_pipeline.transformer,
            args.regional_ip_soft_bias,
        )

    reference_image = load_image_tensor(record["reference_image"])
    reference_tissue_mask = load_tissue_mask(record["reference_tissue_mask"])
    reference_nuclei_mask = load_nuclei_mask(record["reference_nuclei_mask"])
    target_tissue_mask = load_tissue_mask(record["target_tissue_mask"])
    target_nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"])
    source_latent_init_image = load_image_tensor(source_path)

    with torch.no_grad():
        reanchored = run_cross_v1_bundle(
            bundle,
            reference_image=reference_image,
            reference_tissue_mask=reference_tissue_mask,
            reference_nuclei_mask=reference_nuclei_mask,
            target_tissue_mask=target_tissue_mask,
            target_nuclei_mask=target_nuclei_mask,
            prompt=prompt,
            source_latent_init_strength=args.post_controlnet_strength,
            source_latent_init_image=source_latent_init_image,
            seed=args.seed,
        ).convert("RGB")

    reanchored_path = output_dir / args.post_controlnet_output_name
    reanchored.save(reanchored_path)
    panel_path = output_dir / "post_controlnet_reanchor_panel.png"
    make_labeled_grid(
        [
            ("target_gt", load_rgb(record["target_image"])),
            ("reference", load_rgb(record["reference_image"])),
            ("rf_source", load_rgb(source_path)),
            ("reanchored", reanchored),
        ],
        thumbnail_size=args.thumbnail_size,
        title="Post RF-Solver ControlNet re-anchor",
    ).save(panel_path)

    del bundle
    release_cuda_memory()
    return {
        "enabled": True,
        "source": str(source_path),
        "source_copy": str(source_copy),
        "num_inference_steps": int(args.post_controlnet_num_inference_steps),
        "guidance_scale": float(args.post_controlnet_guidance_scale),
        "conditioning_scale": float(args.post_controlnet_conditioning_scale),
        "source_latent_init_strength": float(args.post_controlnet_strength),
        "ip_scale": float(args.post_controlnet_ip_scale),
        "regional_ip_soft_bias": (
            float(args.regional_ip_soft_bias)
            if args.regional_ip_soft_bias is not None
            else None
        ),
        "regional_ip_soft_bias_override": soft_bias_override,
        "artifacts": {
            "reanchored": str(reanchored_path),
            "panel": str(panel_path),
        },
        "notes": (
            "This is a post-hoc ControlNet re-anchor, not in-loop RF-Solver "
            "ControlNet guidance. It tests whether a pathology-trained "
            "ControlNet can repair FLUX natural-image artifacts after RF "
            "cross-image denoising."
        ),
    }


def resolve_post_controlnet_source(
    *,
    explicit: Path | None,
    rf_output_dir: Path,
    source: str,
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"--post-controlnet-source-image does not exist: {explicit}")
        return explicit
    candidates_by_name = {
        "cross_kv": [
            "cross_kv_regional_tissue_nuclei.png",
            "cross_kv_regional_tissue.png",
            "cross_kv_regional_nuclei.png",
            "cross_kv.png",
            "cross_kv_global.png",
        ],
        "cross_v_only": ["cross_v_only.png", "cross_v_only_regional_tissue_nuclei.png"],
        "baseline": ["baseline_reconstruction.png"],
    }
    if source == "auto":
        names = [
            *candidates_by_name["cross_kv"],
            *candidates_by_name["cross_v_only"],
            *candidates_by_name["baseline"],
        ]
    else:
        names = candidates_by_name[source]
    for name in names:
        path = rf_output_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find an RF output image for post ControlNet re-anchor. "
        f"Looked in {rf_output_dir} for {names}."
    )


def build_rf_command(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    i0_path: Path,
    rf_output_dir: Path,
    prompt: str,
) -> list[str]:
    if args.rf_stage_mode == "controlnet":
        return build_rf_controlnet_command(
            args=args,
            record=record,
            i0_path=i0_path,
            rf_output_dir=rf_output_dir,
            prompt=prompt,
        )
    return build_rf_pure_flux_command(
        args=args,
        record=record,
        i0_path=i0_path,
        rf_output_dir=rf_output_dir,
        prompt=prompt,
    )


def build_rf_controlnet_command(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    i0_path: Path,
    rf_output_dir: Path,
    prompt: str,
) -> list[str]:
    command = [
        sys.executable,
        str(args.rf_script),
        "--image",
        str(i0_path),
        "--metadata",
        str(args.metadata),
        "--sample-id",
        str(record.get("sample_id")),
        "--output-dir",
        str(rf_output_dir),
        "--source-prompt",
        args.rf_source_prompt if args.rf_source_prompt is not None else prompt,
        "--num-inference-steps",
        str(args.rf_num_inference_steps),
        "--rf-inversion-guidance",
        str(args.rf_inversion_guidance),
        "--rf-baseline-guidance",
        str(args.rf_baseline_guidance),
        "--controlnet-guidance-scale",
        str(args.rf_controlnet_guidance_scale),
        "--controlnet-conditioning-scale",
        str(args.rf_controlnet_conditioning_scale),
        "--controlnet-start-step",
        str(args.rf_controlnet_start_step),
        "--ip-scale",
        str(args.rf_ip_scale),
        "--controlnet-reference-source",
        args.rf_controlnet_reference_source,
        "--kv-inject",
        "--kv-inject-mode",
        args.rf_cross_image_mode,
        "--kv-inject-strength",
        str(args.rf_cross_image_strength),
        "--kv-inject-start-step",
        str(args.rf_kv_inject_start_step),
        "--kv-inject-after-layer",
        str(args.rf_cross_after_layer),
        "--regional-mode",
        args.rf_regional_mode,
        "--name",
        args.rf_name,
        "--device",
        args.rf_device or args.device,
        "--pretrained-model-name-or-path",
        str(args.pretrained_model_name_or_path),
        "--checkpoint",
        str(args.checkpoint),
        "--uni-checkpoint-path",
        str(args.uni_checkpoint_path),
        "--torch-dtype",
        args.torch_dtype,
    ]
    kv_reference_image = resolve_rf_kv_reference_image(args=args, record=record)
    if kv_reference_image is not None:
        command.extend(["--kv-reference-image", str(kv_reference_image)])
    kv_reference_tissue_mask = resolve_rf_kv_reference_tissue_mask(args=args, record=record)
    if kv_reference_tissue_mask is not None:
        command.extend(["--kv-reference-tissue-mask", str(kv_reference_tissue_mask)])
    kv_reference_nuclei_mask = resolve_rf_kv_reference_nuclei_mask(args=args, record=record)
    if kv_reference_nuclei_mask is not None:
        command.extend(["--kv-reference-nuclei-mask", str(kv_reference_nuclei_mask)])
    command.extend(
        [
            "--kv-reference-prompt",
            args.rf_reference_prompt if args.rf_reference_prompt is not None else prompt,
        ]
    )
    if args.rf_controlnet_reference_source == "metadata":
        command.extend(
            [
                "--reference-image",
                str(record["reference_image"]),
                "--reference-tissue-mask",
                str(record["reference_tissue_mask"]),
                "--reference-nuclei-mask",
                str(record["reference_nuclei_mask"]),
            ]
        )
    if args.rf_with_second_order:
        command.append("--with-second-order")
    else:
        command.append("--no-with-second-order")
    if args.rf_save_rf_baseline:
        command.append("--save-rf-baseline")
    else:
        command.append("--no-save-rf-baseline")
    if args.rf_solver_root is not None:
        command.extend(["--rf-solver-root", str(args.rf_solver_root)])
    if args.flux_diffusers_root is not None:
        command.extend(["--flux-diffusers-root", str(args.flux_diffusers_root)])
    for option, value in (
        ("--t5-model-path", args.t5_model_path),
        ("--t5-tokenizer-path", args.t5_tokenizer_path),
        ("--clip-model-path", args.clip_model_path),
        ("--clip-tokenizer-path", args.clip_tokenizer_path),
    ):
        if value is not None:
            command.extend([option, str(value)])
    if args.rf_regional_ip_soft_bias is not None:
        command.extend(["--regional-ip-soft-bias", str(args.rf_regional_ip_soft_bias)])
    if args.rf_inject_after_t is not None:
        command.extend(["--kv-inject-after-t", str(args.rf_inject_after_t)])
    if args.rf_kv_protect_target_nuclei:
        command.append("--kv-protect-target-nuclei")
        command.extend(
            [
                "--kv-target-nuclei-inject-scale",
                str(args.rf_kv_target_nuclei_inject_scale),
            ]
        )
    if args.rf_kv_block_ref_nuclei_to_target_non_nuclei:
        command.append("--kv-block-ref-nuclei-to-target-non-nuclei")
    if args.rf_kv_protect_target_nuclei or args.rf_kv_block_ref_nuclei_to_target_non_nuclei:
        command.extend(
            [
                "--kv-nuclei-occupancy-dilate-px",
                str(args.rf_kv_nuclei_occupancy_dilate_px),
                "--kv-nuclei-occupancy-min-pixels",
                str(args.rf_kv_nuclei_occupancy_min_pixels),
                "--kv-nuclei-occupancy-min-fraction",
                str(args.rf_kv_nuclei_occupancy_min_fraction),
            ]
        )
    if args.rf_offload:
        command.append("--rf-offload")
    if args.rf_save_feature_debug or args.rf_debug_features:
        command.append("--kv-save-feature-debug")
    if args.rf_fail_on_threshold:
        command.append("--fail-on-threshold")
    if args.rf_allow_text_encoder_download:
        command.append("--allow-text-encoder-download")
    return command


def prepare_rf_kv_reference(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve_rf_kv_reference_image(args=args, record=record)
    if source_path is None:
        raise ValueError("Could not resolve a K/V reference image to preprocess.")
    nuclei_mask_path = (
        args.rf_kv_reference_nuclei_mask
        if args.rf_kv_reference_nuclei_mask is not None
        else Path(str(record["reference_nuclei_mask"]))
    )
    if not source_path.exists():
        raise FileNotFoundError(f"K/V reference image does not exist: {source_path}")
    if not nuclei_mask_path.exists():
        raise FileNotFoundError(f"K/V reference nuclei mask does not exist: {nuclei_mask_path}")

    reference = load_rgb(source_path)
    nuclei_mask = load_label_mask_array(nuclei_mask_path, size=reference.size)
    inpaint_mask = (nuclei_mask != 0).astype(np.uint8) * 255
    if int(args.rf_kv_inpaint_dilate) > 0:
        inpaint_mask = dilate_binary_mask(inpaint_mask, int(args.rf_kv_inpaint_dilate))
    mask_path = output_dir / "kv_reference_inpaint_nuclei_mask.png"
    Image.fromarray(inpaint_mask, mode="L").save(mask_path)

    if args.rf_kv_reference_preprocess == "inpaint_ref_nuclei":
        inpainted, backend = inpaint_rgb_image(
            reference,
            inpaint_mask,
            radius=float(args.rf_kv_inpaint_radius),
            method=args.rf_kv_inpaint_method,
        )
    else:
        raise ValueError(f"Unsupported K/V reference preprocess: {args.rf_kv_reference_preprocess}")

    output_path = output_dir / "kv_reference_inpaint_ref_nuclei.png"
    inpainted.save(output_path)
    args.rf_reference_image = output_path
    panel_path = output_dir / "kv_reference_preprocess_panel.png"
    make_labeled_grid(
        [
            ("original_ref", reference),
            ("nuclei_mask", Image.fromarray(inpaint_mask, mode="L").convert("RGB")),
            ("tissue_only_ref", inpainted),
        ],
        thumbnail_size=192,
        title="K/V reference preprocess: inpaint reference nuclei",
        columns=3,
    ).save(panel_path)
    print(
        "Prepared tissue-only K/V reference by inpainting ref nuclei: "
        f"{output_path} backend={backend}"
    )
    return {
        "mode": args.rf_kv_reference_preprocess,
        "applied": True,
        "source_image": str(source_path),
        "source_nuclei_mask": str(nuclei_mask_path),
        "inpaint_mask": str(mask_path),
        "output_image": str(output_path),
        "panel": str(panel_path),
        "backend": backend,
        "radius": float(args.rf_kv_inpaint_radius),
        "dilate": int(args.rf_kv_inpaint_dilate),
        "method": args.rf_kv_inpaint_method,
        "masked_pixel_count": int((inpaint_mask > 0).sum()),
    }


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return mask
    try:
        import cv2

        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel, iterations=1)
    except Exception:
        image = Image.fromarray(mask, mode="L")
        for _ in range(int(radius)):
            image = image.filter(ImageFilter.MaxFilter(3))
        return np.asarray(image, dtype=np.uint8)


def inpaint_rgb_image(
    image: Image.Image,
    mask: np.ndarray,
    *,
    radius: float,
    method: str,
) -> tuple[Image.Image, str]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not np.any(mask_u8 > 0):
        return image.convert("RGB"), "none_empty_mask"
    try:
        import cv2

        flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        output_bgr = cv2.inpaint(bgr, mask_u8, float(radius), flag)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb, mode="RGB"), f"opencv_{method}"
    except Exception:
        mask_bool = mask_u8 > 0
        blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(radius, 1.0)))
        output = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        output[mask_bool] = np.asarray(blurred, dtype=np.uint8)[mask_bool]
        return Image.fromarray(output, mode="RGB"), "pil_blur_fallback"


def build_rf_pure_flux_command(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    i0_path: Path,
    rf_output_dir: Path,
    prompt: str,
) -> list[str]:
    command = [
        sys.executable,
        str(args.rf_script),
        "--cross-image",
        "--image",
        str(i0_path),
        "--metadata",
        str(args.metadata),
        "--sample-id",
        str(record.get("sample_id")),
        "--output-dir",
        str(rf_output_dir),
        "--source-prompt",
        args.rf_source_prompt if args.rf_source_prompt is not None else prompt,
        "--reference-prompt",
        args.rf_reference_prompt if args.rf_reference_prompt is not None else prompt,
        "--num-inference-steps",
        str(args.rf_num_inference_steps),
        "--guidance",
        str(args.rf_guidance),
        "--name",
        args.rf_name,
        "--device",
        args.rf_device or args.device,
        "--cross-image-mode",
        args.rf_cross_image_mode,
        "--cross-image-strength",
        str(args.rf_cross_image_strength),
        "--inject-steps",
        str(args.rf_inject_steps),
        "--inject-after-t",
        str(1.0 if args.rf_inject_after_t is None else args.rf_inject_after_t),
        "--cross-after-layer",
        str(args.rf_cross_after_layer),
        "--regional-mode",
        args.rf_regional_mode,
    ]
    if args.rf_target_prompt is not None:
        command.extend(["--target-prompt", args.rf_target_prompt])
    if args.rf_with_second_order:
        command.append("--with-second-order")
    else:
        command.append("--no-with-second-order")
    if args.rf_solver_root is not None:
        command.extend(["--rf-solver-root", str(args.rf_solver_root)])
    if args.flux_diffusers_root is not None:
        command.extend(["--flux-diffusers-root", str(args.flux_diffusers_root)])
    for option, value in (
        ("--t5-model-path", args.t5_model_path),
        ("--t5-tokenizer-path", args.t5_tokenizer_path),
        ("--clip-model-path", args.clip_model_path),
        ("--clip-tokenizer-path", args.clip_tokenizer_path),
    ):
        if value is not None:
            command.extend([option, str(value)])
    if args.rf_reference_image is not None:
        command.extend(["--reference-image", str(args.rf_reference_image)])
    if args.rf_reference_sample_id is not None:
        command.extend(["--reference-sample-id", args.rf_reference_sample_id])
    if args.rf_reference_metadata_index is not None:
        command.extend(["--reference-metadata-index", str(args.rf_reference_metadata_index)])
    if (
        args.rf_reference_sample_id is not None
        or args.rf_reference_metadata_index is not None
        or args.rf_auto_reference_by_texture
    ):
        command.extend(
            [
                "--reference-record-image-field",
                args.rf_reference_record_image_field,
            ]
        )
    if args.rf_auto_reference_by_texture:
        command.append("--auto-reference-by-texture")
        command.extend(
            [
                "--auto-reference-max-candidates",
                str(args.rf_auto_reference_max_candidates),
                "--auto-reference-rank",
                str(args.rf_auto_reference_rank),
            ]
        )
        command.append(
            "--auto-reference-same-dataset"
            if args.rf_auto_reference_same_dataset
            else "--no-auto-reference-same-dataset"
        )
        command.append(
            "--auto-reference-different-case"
            if args.rf_auto_reference_different_case
            else "--no-auto-reference-different-case"
        )
    if args.rf_offload:
        command.append("--offload")
    if args.rf_save_feature_debug:
        command.append("--save-feature-debug")
    if args.rf_debug_features:
        command.append("--debug-features")
    if args.rf_fail_on_threshold:
        command.append("--fail-on-threshold")
    if args.rf_allow_text_encoder_download:
        command.append("--allow-text-encoder-download")
    return command


def resolve_rf_kv_reference_image(*, args: argparse.Namespace, record: dict[str, Any]) -> Path | None:
    if args.rf_reference_image is not None:
        return args.rf_reference_image
    if args.rf_reference_sample_id is not None or args.rf_reference_metadata_index is not None:
        reference_record, _ = select_metadata_record(
            args.metadata,
            args.rf_reference_sample_id,
            args.rf_reference_metadata_index if args.rf_reference_metadata_index is not None else 0,
        )
        field = args.rf_reference_record_image_field
        value = reference_record.get(field)
        if not value:
            raise KeyError(f"Selected RF reference record has no {field!r} field.")
        return Path(str(value))
    if args.rf_auto_reference_by_texture:
        raise ValueError(
            "--rf-auto-reference-by-texture is currently available only with "
            "--rf-stage-mode pure_flux. Pass --rf-reference-image for the "
            "ControlNet-CIA stage."
        )
    value = record.get("reference_image")
    return Path(str(value)) if value else None


def resolve_rf_kv_reference_tissue_mask(*, args: argparse.Namespace, record: dict[str, Any]) -> Path | None:
    if args.rf_kv_reference_tissue_mask is not None:
        return args.rf_kv_reference_tissue_mask
    reference_record = selected_rf_reference_record(args=args)
    if reference_record is not None:
        field = "target_tissue_mask" if args.rf_reference_record_image_field == "target_image" else "reference_tissue_mask"
        value = reference_record.get(field)
        return Path(str(value)) if value else None
    value = record.get("reference_tissue_mask")
    return Path(str(value)) if value else None


def resolve_rf_kv_reference_nuclei_mask(*, args: argparse.Namespace, record: dict[str, Any]) -> Path | None:
    if args.rf_kv_reference_nuclei_mask is not None:
        return args.rf_kv_reference_nuclei_mask
    reference_record = selected_rf_reference_record(args=args)
    if reference_record is not None:
        field = "target_nuclei_mask" if args.rf_reference_record_image_field == "target_image" else "reference_nuclei_mask"
        value = reference_record.get(field)
        return Path(str(value)) if value else None
    value = record.get("reference_nuclei_mask")
    return Path(str(value)) if value else None


def selected_rf_reference_record(*, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.rf_reference_sample_id is None and args.rf_reference_metadata_index is None:
        return None
    reference_record, _ = select_metadata_record(
        args.metadata,
        args.rf_reference_sample_id,
        args.rf_reference_metadata_index if args.rf_reference_metadata_index is not None else 0,
    )
    return reference_record


def apply_controlnet_stain_match(
    *,
    args: argparse.Namespace,
    i0: Image.Image,
    record: dict[str, Any],
    output_dir: Path,
) -> tuple[Image.Image, dict[str, Any]]:
    mode = str(args.controlnet_color_match or "none").strip().lower()
    if mode in {"", "none", "off", "false"}:
        return i0.convert("RGB"), {
            "mode": "none",
            "applied": False,
            "input": "controlnet_i0_raw.png",
            "output": "controlnet_i0.png",
        }
    if mode in {"hed", "hd"}:
        # The old HED code is an augmentation path; for deterministic matching,
        # use the existing Macenko H&E stain transfer implementation.
        normalized_mode = "macenko"
    else:
        normalized_mode = mode

    if normalized_mode == "lab":
        matched = match_image_color_to_reference(
            i0,
            Image.open(record["reference_image"]).convert("RGB"),
        )
        matched = blend_images(
            base=i0,
            edited=matched,
            strength=args.controlnet_color_match_strength,
        )
        matched_path = output_dir / "controlnet_i0_stain_matched_lab.png"
        matched.save(matched_path)
        return matched, {
            "mode": mode,
            "normalized_mode": normalized_mode,
            "applied": True,
            "scope": "global",
            "strength": float(args.controlnet_color_match_strength),
            "matched_path": str(matched_path),
            "reference_image": record.get("reference_image"),
        }

    if normalized_mode != "macenko":
        raise ValueError(
            "--controlnet-color-match must be one of: none, lab, macenko, hed, hd"
        )

    reference = load_rgb(record["reference_image"])
    target_mask = None
    reference_mask = None
    target_mask_path = None
    reference_mask_path = None
    scope = str(args.controlnet_color_match_scope or "region").strip().lower()
    if scope == "region":
        target_mask = load_composite_stain_mask(
            record["target_tissue_mask"],
            record["target_nuclei_mask"],
            size=i0.size,
        )
        reference_mask = load_composite_stain_mask(
            record["reference_tissue_mask"],
            record["reference_nuclei_mask"],
            size=reference.size,
        )
        target_mask_path = output_dir / "target_stain_match_composite_mask.png"
        reference_mask_path = output_dir / "reference_stain_match_composite_mask.png"
        save_label_mask_image(target_mask, target_mask_path)
        save_label_mask_image(reference_mask, reference_mask_path)
        matched_array = macenko_stain_transfer_by_mask_local(
            np.asarray(i0.convert("RGB"), dtype=np.uint8),
            np.asarray(reference.convert("RGB"), dtype=np.uint8),
            target_mask,
            reference_mask,
            background_label=args.controlnet_color_match_background_label,
            fallback=args.controlnet_color_match_fallback,
            io=args.controlnet_macenko_io,
            beta=args.controlnet_macenko_beta,
            alpha=args.controlnet_macenko_alpha,
        )
    else:
        target_tissue = load_label_mask_array(record["target_tissue_mask"], size=i0.size)
        reference_tissue = load_label_mask_array(
            record["reference_tissue_mask"],
            size=reference.size,
        )
        background_label = int(args.controlnet_color_match_background_label)
        matched_array = macenko_stain_transfer_local(
            np.asarray(i0.convert("RGB"), dtype=np.uint8),
            np.asarray(reference.convert("RGB"), dtype=np.uint8),
            source_mask=target_tissue != background_label,
            reference_mask=reference_tissue != background_label,
            io=args.controlnet_macenko_io,
            beta=args.controlnet_macenko_beta,
            alpha=args.controlnet_macenko_alpha,
        )
    matched = Image.fromarray(matched_array, mode="RGB").convert("RGB")
    matched = blend_images(
        base=i0,
        edited=matched,
        strength=args.controlnet_color_match_strength,
    )
    matched_path = output_dir / f"controlnet_i0_stain_matched_{mode}.png"
    matched.save(matched_path)
    print(
        "Applied ControlNet I0 stain match "
        f"mode={mode} normalized={normalized_mode} scope={scope} "
        f"strength={args.controlnet_color_match_strength:g}"
    )
    return matched, {
        "mode": mode,
        "normalized_mode": normalized_mode,
        "applied": True,
        "scope": scope,
        "strength": float(args.controlnet_color_match_strength),
        "background_label": int(args.controlnet_color_match_background_label),
        "fallback": args.controlnet_color_match_fallback,
        "macenko_io": float(args.controlnet_macenko_io),
        "macenko_beta": float(args.controlnet_macenko_beta),
        "macenko_alpha": float(args.controlnet_macenko_alpha),
        "matched_path": str(matched_path),
        "reference_image": record.get("reference_image"),
        "target_mask": str(target_mask_path) if target_mask_path is not None else None,
        "reference_mask": str(reference_mask_path) if reference_mask_path is not None else None,
    }


def blend_images(*, base: Image.Image, edited: Image.Image, strength: float) -> Image.Image:
    alpha = float(np.clip(strength, 0.0, 1.0))
    if alpha >= 1.0:
        return edited.convert("RGB")
    if alpha <= 0.0:
        return base.convert("RGB")
    base_array = np.asarray(base.convert("RGB"), dtype=np.float32)
    edited_array = np.asarray(edited.convert("RGB"), dtype=np.float32)
    if base_array.shape != edited_array.shape:
        edited_array = np.asarray(
            edited.convert("RGB").resize(base.size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
    output = base_array * (1.0 - alpha) + edited_array * alpha
    return Image.fromarray(np.clip(output.round(), 0, 255).astype(np.uint8), mode="RGB")


def match_image_color_to_reference(source: Image.Image, reference: Image.Image) -> Image.Image:
    from skimage.color import lab2rgb, rgb2lab

    source_rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    reference_rgb = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    source_lab = rgb2lab(source_rgb).astype(np.float32)
    reference_lab = rgb2lab(reference_rgb).astype(np.float32)
    source_mask = tissue_mask_from_rgb(source_rgb)
    reference_mask = tissue_mask_from_rgb(reference_rgb)
    if not np.any(source_mask) or not np.any(reference_mask):
        return source.convert("RGB")

    matched_lab = source_lab.copy()
    for channel in range(3):
        source_values = source_lab[..., channel][source_mask]
        reference_values = reference_lab[..., channel][reference_mask]
        source_std = float(source_values.std())
        reference_std = float(reference_values.std())
        matched_lab[..., channel][source_mask] = (
            (source_values - float(source_values.mean()))
            * (reference_std / max(source_std, 1e-6))
            + float(reference_values.mean())
        )
    matched_rgb = np.clip(lab2rgb(matched_lab), 0.0, 1.0)
    output = source_rgb.copy()
    output[source_mask] = matched_rgb[source_mask]
    return Image.fromarray((output * 255.0).round().astype(np.uint8), mode="RGB")


def tissue_mask_from_rgb(rgb_float: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    return rgb_float.mean(axis=-1) < threshold


def pil_to_tensor(image: Image.Image):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)


def load_label_mask_array(path: str | Path, *, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        mask = ImageOps.exif_transpose(image)
        if size is not None and mask.size != tuple(size):
            mask = mask.resize(size, Image.Resampling.NEAREST)
        array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.int64, copy=False)


def load_composite_stain_mask(
    tissue_mask_path: str | Path,
    nuclei_mask_path: str | Path,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    tissue = load_label_mask_array(tissue_mask_path, size=size).copy()
    nuclei = load_label_mask_array(nuclei_mask_path, size=size)
    nuclei_pixels = nuclei != 0
    tissue[nuclei_pixels] = nuclei[nuclei_pixels] + NUCLEI_STAIN_LABEL_OFFSET
    return tissue


def save_label_mask_image(mask, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(mask):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.size == 0 or int(array.max(initial=0)) <= 255:
        image_array = array.astype(np.uint8, copy=False)
    else:
        image_array = array.astype(np.uint16, copy=False)
    Image.fromarray(image_array).save(path)


def macenko_stain_transfer_by_mask_local(
    source: np.ndarray,
    reference: np.ndarray,
    target_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    background_label: int = 0,
    fallback: str = "pooled",
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
    min_region_pixels: int = 10,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    output = np.asarray(source, dtype=np.uint8).copy()
    target_mask = np.asarray(target_mask)
    reference_mask = np.asarray(reference_mask)
    pooled_source = target_mask != int(background_label)
    pooled_reference = reference_mask != int(background_label)
    he_source = estimate_macenko_stain_matrix_local(
        source,
        mask=pooled_source,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    he_reference = estimate_macenko_stain_matrix_local(
        reference,
        mask=pooled_reference,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    conc_source = macenko_concentrations_local(source, he_source, io=io)
    conc_reference = macenko_concentrations_local(reference, he_reference, io=io)
    target_labels = [
        int(label)
        for label in np.unique(target_mask)
        if int(label) != int(background_label)
    ]
    reference_labels = {
        int(label)
        for label in np.unique(reference_mask)
        if int(label) != int(background_label)
    }
    pooled_reference = reference_mask != int(background_label)
    fallback_mode = str(fallback or "pooled").strip().lower()
    for label in sorted(target_labels):
        source_region = target_mask == int(label)
        if int(source_region.sum()) < int(min_region_pixels):
            continue
        if label in reference_labels and int((reference_mask == label).sum()) >= int(min_region_pixels):
            reference_region = reference_mask == label
        elif fallback_mode == "pooled" and int(pooled_reference.sum()) >= int(min_region_pixels):
            reference_region = pooled_reference
        else:
            continue
        transferred = macenko_apply_concentration_match_local(
            source,
            conc_source,
            conc_reference,
            he_reference,
            source_mask=source_region,
            reference_mask=reference_region,
            io=io,
        )
        output[source_region] = transferred[source_region]
    return output


def macenko_apply_concentration_match_local(
    source: np.ndarray,
    conc_source: np.ndarray,
    conc_reference: np.ndarray,
    reference_stain_matrix: np.ndarray,
    *,
    source_mask: np.ndarray,
    reference_mask: np.ndarray,
    io: float = 240.0,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    h, w, _ = source.shape
    source_select = valid_bool_mask(source_mask, (h, w))
    if source_select is None:
        return source.copy()
    reference_select = np.asarray(reference_mask, dtype=bool)
    if not np.any(reference_select):
        return source.copy()
    source_flat_mask = source_select.reshape(-1)
    reference_flat_mask = reference_select.reshape(-1)
    if int(source_flat_mask.sum()) < 1 or int(reference_flat_mask.sum()) < 1:
        return source.copy()
    max_source = np.percentile(conc_source[source_flat_mask], 99, axis=0)
    max_reference = np.percentile(conc_reference[reference_flat_mask], 99, axis=0)
    max_source = np.where(max_source < 1e-6, 1e-6, max_source)
    region_conc = conc_source[source_flat_mask] * (max_reference / max_source)[None, :]
    region_rgb = od_to_rgb_local(region_conc @ reference_stain_matrix, io=io)
    output = source.copy().reshape(-1, 3)
    output[source_flat_mask] = region_rgb
    return output.reshape(h, w, 3)


def macenko_stain_transfer_local(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    h, w, _ = source.shape
    source_select = valid_bool_mask(source_mask, (h, w))
    reference_select = valid_bool_mask(reference_mask, reference.shape[:2])
    he_source = estimate_macenko_stain_matrix_local(
        source,
        mask=source_select,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    he_reference = estimate_macenko_stain_matrix_local(
        reference,
        mask=reference_select,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    conc_source = macenko_concentrations_local(source, he_source, io=io)
    conc_reference = macenko_concentrations_local(reference, he_reference, io=io)
    source_flat_mask = (
        source_select.reshape(-1)
        if source_select is not None
        else np.ones((h * w,), dtype=bool)
    )
    reference_flat_mask = (
        reference_select.reshape(-1)
        if reference_select is not None
        else np.ones((reference.shape[0] * reference.shape[1],), dtype=bool)
    )
    if int(source_flat_mask.sum()) < 1 or int(reference_flat_mask.sum()) < 1:
        return source.copy()
    max_source = np.percentile(conc_source[source_flat_mask], 99, axis=0)
    max_reference = np.percentile(conc_reference[reference_flat_mask], 99, axis=0)
    max_source = np.where(max_source < 1e-6, 1e-6, max_source)
    conc_matched = conc_source * (max_reference / max_source)[None, :]
    od_new = conc_matched @ he_reference
    rgb_new = od_to_rgb_local(od_new.reshape(h, w, 3), io=io)
    output = source.copy()
    if source_select is None:
        output = rgb_new
    else:
        output[source_select] = rgb_new[source_select]
    return output


def rgb_to_od_local(image: np.ndarray, io: float = 240.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return -np.log((image + 1.0) / float(io))


def od_to_rgb_local(od: np.ndarray, io: float = 240.0) -> np.ndarray:
    rgb = float(io) * np.exp(-np.asarray(od, dtype=np.float64))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def estimate_macenko_stain_matrix_local(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    od = rgb_to_od_local(image, io=io).reshape(-1, 3)
    if mask is not None:
        od = od[np.asarray(mask, dtype=bool).reshape(-1)]
    od = od[np.all(np.isfinite(od), axis=1)]
    if od.shape[0] < 3:
        return default_he_matrix_local()
    od = np.clip(od, 0.0, None)
    stain_strength = np.linalg.norm(od, axis=1)
    od_hat = od[stain_strength > float(beta)]
    if od_hat.shape[0] < 10:
        relaxed_threshold = max(float(beta) * 0.25, 1e-6)
        od_hat = od[stain_strength > relaxed_threshold]
    if od_hat.shape[0] < 3:
        return default_he_matrix_local()
    try:
        cov = np.cov(od_hat.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
            return default_he_matrix_local()
        order = np.argsort(eigvals)[::-1][:2]
        v = eigvecs[:, order]
        if v[0, 0] < 0:
            v[:, 0] *= -1
        if v[0, 1] < 0:
            v[:, 1] *= -1
        projection = od_hat @ v
        phi = np.arctan2(projection[:, 1], projection[:, 0])
        min_phi = np.percentile(phi, float(alpha))
        max_phi = np.percentile(phi, 100.0 - float(alpha))
        v1 = v @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = v @ np.array([np.cos(max_phi), np.sin(max_phi)])
        he = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
        he = np.clip(he, 1e-6, None)
        norms = np.linalg.norm(he, axis=1, keepdims=True)
        if np.any(norms < 1e-8) or not np.all(np.isfinite(he)):
            return default_he_matrix_local()
        return he / norms
    except np.linalg.LinAlgError:
        return default_he_matrix_local()


def macenko_concentrations_local(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    *,
    io: float = 240.0,
) -> np.ndarray:
    od = rgb_to_od_local(image, io=io).reshape(-1, 3)
    concentrations = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0].T
    return np.clip(concentrations, 0.0, None)


def default_he_matrix_local() -> np.ndarray:
    he = np.array(
        [
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
        ],
        dtype=np.float64,
    )
    return he / np.linalg.norm(he, axis=1, keepdims=True)


def valid_bool_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    value = np.asarray(mask, dtype=bool)
    if value.shape != tuple(shape):
        return None
    if not np.any(value):
        return None
    return value


def save_controlnet_panel(
    *,
    reference: Image.Image,
    target: Image.Image,
    i0_raw: Image.Image,
    i0: Image.Image,
    output_path: Path,
    thumbnail_size: int,
    title: str,
) -> None:
    images = [
        ("target_gt", target),
        ("reference", reference),
        ("controlnet_i0_raw", i0_raw),
        ("controlnet_i0", i0),
    ]
    make_labeled_grid(images, thumbnail_size=thumbnail_size, title=title).save(output_path)


def save_final_panel(
    *,
    output_dir: Path,
    target_path: Path,
    reference_path: Path,
    i0_path: Path,
    rf_output_dir: Path,
    post_reanchor_path: Path | None = None,
    thumbnail_size: int,
) -> Path | None:
    if not rf_output_dir.exists():
        return None
    images: list[tuple[str, Image.Image]] = [
        ("target_gt", load_rgb(target_path)),
        ("reference", load_rgb(reference_path)),
        ("controlnet_i0", load_rgb(i0_path)),
    ]
    for label, filename in (
        ("rf_baseline", "rf_baseline_reconstruction.png"),
        ("controlnet_recon", "controlnet_reconstruction.png"),
        ("controlnet_kv", "controlnet_kv_reconstruction.png"),
        ("rf_baseline", "baseline_reconstruction.png"),
        ("cross_kv", "cross_kv.png"),
        ("cross_v_only", "cross_v_only.png"),
        ("cross_kv_global", "cross_kv_global.png"),
        ("cross_regional_tissue", "cross_kv_regional_tissue.png"),
        ("cross_regional_nuclei", "cross_kv_regional_nuclei.png"),
        ("cross_regional", "cross_kv_regional_tissue_nuclei.png"),
    ):
        path = rf_output_dir / filename
        if path.exists():
            images.append((label, load_rgb(path)))
    if post_reanchor_path is not None and post_reanchor_path.exists():
        images.append(("post_controlnet", load_rgb(post_reanchor_path)))
    if len(images) <= 3:
        return None
    output_path = output_dir / "controlnet_i0_then_rf_solver_panel.png"
    make_labeled_grid(
        images,
        thumbnail_size=thumbnail_size,
        title="ControlNet I0 -> RF-Solver/ControlNet CIA",
    ).save(output_path)
    return output_path


def save_cia_postprocess_diagnostics(
    *,
    output_dir: Path,
    target_path: Path,
    reference_path: Path,
    i0_path: Path,
    rf_output_dir: Path,
    thumbnail_size: int,
) -> dict[str, Any]:
    cia_path = resolve_cia_output_path(rf_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = load_rgb(target_path)
    reference = load_rgb(reference_path)
    i0 = load_rgb(i0_path)
    cia = load_rgb(cia_path)
    diff = make_diff_image_local(i0, cia)

    diff_path = output_dir / "i0_vs_cia_diff.png"
    diff.save(diff_path)
    panel_path = output_dir / "i0_vs_cia_panel.png"
    make_labeled_grid(
        [
            ("target_gt", target),
            ("reference", reference),
            ("controlnet_i0", i0),
            ("controlnet_cia", cia),
            ("abs_diff_i0_cia", diff),
        ],
        thumbnail_size=thumbnail_size,
        title="I0 vs ControlNet-CIA",
        columns=5,
    ).save(panel_path)

    metrics = image_metrics_local(i0, cia)
    metrics.update(
        {
            "selected_cia_output": str(cia_path),
            "rf_output_dir": str(rf_output_dir),
            "artifacts": {
                "diff": str(diff_path),
                "panel": str(panel_path),
            },
            "interpretation": (
                "These metrics compare generated I0 to the CIA output. They are "
                "diagnostics for how much the postprocess changed I0, not a "
                "target-GT reconstruction gate."
            ),
        }
    )
    metrics_path = output_dir / "cia_postprocess_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf8",
    )
    print(
        "Saved I0-vs-CIA diagnostics: "
        f"cia={cia_path.name} mae={metrics['mae']:.6f} ssim={metrics.get('ssim')}"
    )
    return {
        "selected_cia_output": str(cia_path),
        "metrics": str(metrics_path),
        "panel": str(panel_path),
        "diff": str(diff_path),
        "mae": metrics["mae"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
    }


def resolve_cia_output_path(rf_output_dir: Path) -> Path:
    candidates = [
        "controlnet_kv_reconstruction.png",
        "controlnet_reconstruction.png",
        "cross_kv_regional_tissue_nuclei.png",
        "cross_kv_regional_tissue.png",
        "cross_kv_regional_nuclei.png",
        "cross_kv.png",
        "cross_kv_global.png",
        "cross_v_only.png",
    ]
    for name in candidates:
        path = rf_output_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a CIA output image. Looked in "
        f"{rf_output_dir} for {candidates}."
    )


def make_diff_image_local(a: Image.Image, b: Image.Image) -> Image.Image:
    a_array = np.asarray(a.convert("RGB"), dtype=np.float32)
    b_image = b.convert("RGB")
    if b_image.size != a.size:
        b_image = b_image.resize(a.size, Image.Resampling.BICUBIC)
    b_array = np.asarray(b_image, dtype=np.float32)
    diff = np.abs(a_array - b_array)
    diff = np.clip(diff * 4.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def image_metrics_local(a: Image.Image, b: Image.Image) -> dict[str, Any]:
    a_array = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    b_image = b.convert("RGB")
    if b_image.size != a.size:
        b_image = b_image.resize(a.size, Image.Resampling.BICUBIC)
    b_array = np.asarray(b_image, dtype=np.float32) / 255.0
    mse = float(np.mean((a_array - b_array) ** 2))
    mae = float(np.mean(np.abs(a_array - b_array)))
    max_abs = float(np.max(np.abs(a_array - b_array)))
    psnr: float | str
    psnr = "inf" if mse <= 0.0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    ssim: float | None = None
    try:
        from skimage.metrics import structural_similarity

        ssim = float(
            structural_similarity(
                a_array,
                b_array,
                channel_axis=2,
                data_range=1.0,
            )
        )
    except Exception:
        ssim = None
    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "psnr": psnr,
        "ssim": ssim,
    }


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_labeled_grid(
    images: list[tuple[str, Image.Image]],
    *,
    thumbnail_size: int,
    title: str,
    columns: int | None = None,
) -> Image.Image:
    if columns is None:
        columns = min(len(images), 4)
    rows = int(math.ceil(len(images) / columns))
    title_h = 28
    label_h = 28
    width = columns * thumbnail_size
    height = title_h + rows * (thumbnail_size + label_h)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title[:180], fill=(0, 0, 0))
    for index, (label, image) in enumerate(images):
        row = index // columns
        col = index % columns
        x = col * thumbnail_size
        y = title_h + row * (thumbnail_size + label_h)
        panel.paste(thumbnail(image, thumbnail_size), (x, y))
        draw.text((x + 6, y + thumbnail_size + 7), label, fill=(0, 0, 0))
    return panel


def thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def write_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    record: dict[str, Any],
    metadata_index: int,
    prompt: str,
    controlnet_summary: dict[str, Any],
    i0_raw_path: Path,
    i0_path: Path,
    rf_command: list[str],
    rf_output_dir: Path,
    rf_result: dict[str, Any],
    started_at: float,
) -> Path:
    summary = {
        "mode": (
            "controlnet_i0_then_rf_inversion_controlnet_cia"
            if args.rf_stage_mode == "controlnet"
            else "controlnet_i0_then_rf_solver_cross"
        ),
        "metadata": str(args.metadata),
        "metadata_index": int(metadata_index),
        "sample_id": record.get("sample_id"),
        "reference_sample_id": record.get("reference_sample_id"),
        "case_id": record.get("case_id"),
        "dataset": record.get("dataset"),
        "prompt": prompt,
        "inputs": {
            "target_image": record.get("target_image"),
            "reference_image": record.get("reference_image"),
            "target_tissue_mask": record.get("target_tissue_mask"),
            "target_nuclei_mask": record.get("target_nuclei_mask"),
            "reference_tissue_mask": record.get("reference_tissue_mask"),
            "reference_nuclei_mask": record.get("reference_nuclei_mask"),
        },
        "controlnet": controlnet_summary,
        "rf_solver": {
            "stage_mode": args.rf_stage_mode,
            "script": str(args.rf_script),
            "output_dir": str(rf_output_dir),
            "command": rf_command,
            "command_shell": " ".join(shell_quote(part) for part in rf_command),
            "result": rf_result,
            "num_inference_steps": int(args.rf_num_inference_steps),
            "with_second_order": bool(args.rf_with_second_order),
            "guidance": float(args.rf_guidance),
            "inversion_guidance": float(args.rf_inversion_guidance),
            "baseline_guidance": float(args.rf_baseline_guidance),
            "cross_image_mode": args.rf_cross_image_mode,
            "cross_image_strength": float(args.rf_cross_image_strength),
            "inject_steps": int(args.rf_inject_steps),
            "kv_inject_start_step": int(args.rf_kv_inject_start_step),
            "inject_after_t": (
                None if args.rf_inject_after_t is None else float(args.rf_inject_after_t)
            ),
            "cross_after_layer": int(args.rf_cross_after_layer),
            "regional_mode": args.rf_regional_mode,
            "kv_protect_target_nuclei": bool(args.rf_kv_protect_target_nuclei),
            "kv_target_nuclei_inject_scale": float(args.rf_kv_target_nuclei_inject_scale),
            "kv_block_ref_nuclei_to_target_non_nuclei": bool(
                args.rf_kv_block_ref_nuclei_to_target_non_nuclei
            ),
            "kv_nuclei_occupancy": {
                "dilate_px": int(args.rf_kv_nuclei_occupancy_dilate_px),
                "min_pixels": int(args.rf_kv_nuclei_occupancy_min_pixels),
                "min_fraction": float(args.rf_kv_nuclei_occupancy_min_fraction),
            },
            "controlnet": {
                "guidance_scale": float(args.rf_controlnet_guidance_scale),
                "conditioning_scale": float(args.rf_controlnet_conditioning_scale),
                "start_step": int(args.rf_controlnet_start_step),
                "reference_source": args.rf_controlnet_reference_source,
                "ip_scale": float(args.rf_ip_scale),
                "regional_ip_soft_bias": args.rf_regional_ip_soft_bias,
            },
            "reference_override": {
                "reference_image": str(args.rf_reference_image)
                if args.rf_reference_image is not None
                else None,
                "reference_sample_id": args.rf_reference_sample_id,
                "reference_metadata_index": args.rf_reference_metadata_index,
                "auto_reference_by_texture": bool(args.rf_auto_reference_by_texture),
                "reference_record_image_field": args.rf_reference_record_image_field,
            },
            "kv_reference_preprocess": rf_result.get("kv_reference_preprocess"),
        },
        "artifacts": {
            "i0_source": args.i0_source,
            "controlnet_i0_raw": str(i0_raw_path),
            "controlnet_i0": str(i0_path),
            "controlnet_panel": str(output_dir / "controlnet_i0" / "controlnet_i0_panel.png"),
            "final_panel": rf_result.get("final_panel"),
        },
        "runtime_seconds": round(time.perf_counter() - started_at, 3),
        "notes": (
            "I0 is generated by the pathology-trained Cross V1 ControlNet. "
            "In the default stage_mode=controlnet path, Stage 2 does pure FLUX "
            "RF inversion of I0, then denoises from I0 zT with ControlNet "
            "residuals while injecting reference K/V from the selected "
            "appearance reference."
        ),
    }
    summary_path = output_dir / "controlnet_i0_then_rf_solver_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf8",
    )
    return summary_path


def shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/.:=,")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())

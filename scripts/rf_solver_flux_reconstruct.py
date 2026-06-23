#!/usr/bin/env python
"""RF-Solver FLUX inversion/reconstruction gate for pathology patches.

This runner intentionally bypasses RF-Solver-Edit's watermarking and NSFW
post-processing so the saved reconstruction can be compared directly with the
input image.

Example:
    RF_SOLVER_ROOT=/data/wqx/RF-Solver-Edit/FLUX_Image_Edit \
    PYTHONPATH=/data/wqx/RF-Solver-Edit/FLUX_Image_Edit/src:$PYTHONPATH \
    python scripts/rf_solver_flux_reconstruct.py \
        --image /path/to/pathology.png \
        --output-dir phase5_runs/rf_solver_flux_reconstruct/pathology_25 \
        --num-inference-steps 25 \
        --with-second-order \
        --guidance 1.0 \
        --fail-on-threshold
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

os.environ.setdefault(
    "SKIMAGE_DATADIR", str(Path(tempfile.gettempdir()) / "skimage-data")
)


DEFAULT_SOURCE_PROMPT = (
    "hematoxylin and eosin stained pathology tissue microscopy image"
)
DEFAULT_PSNR_THRESHOLD = 25.0
DEFAULT_SSIM_THRESHOLD = 0.85
DEFAULT_CROSS_INJECT_STEPS = 20
DEFAULT_CROSS_AFTER_LAYER = 20
DEFAULT_TEXT_TOKEN_COUNT = 512
DEFAULT_IMAGE_TOKEN_COUNT = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Invert one pathology image with RF-Solver FLUX and denoise it back "
            "to measure reconstruction fidelity."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Path to the pathology RGB image to reconstruct. If omitted, "
            "--metadata supplies this from --image-field."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where reconstruction artifacts and metrics are written. "
            "Defaults to a metadata sample-specific directory when --metadata is used."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional cross metadata JSON, e.g. phase5_runs/cross_meta/"
            "metadata_cross_val.json. Accepts either a raw list or {'pairs': [...]}."
        ),
    )
    parser.add_argument(
        "--metadata-index",
        type=int,
        default=0,
        help="Record index to use when --metadata is set and --sample-id is omitted.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="sample_id to select from --metadata.",
    )
    parser.add_argument(
        "--image-field",
        choices=("target_image", "reference_image"),
        default="target_image",
        help="Metadata image field to reconstruct. Use target_image for the STEP 1 gate.",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Metadata prompt field used when --source-prompt is omitted.",
    )
    parser.add_argument(
        "--source-prompt",
        default=None,
        help=(
            "Prompt used for inversion. If omitted, uses metadata[prompt] when "
            "--metadata is set, otherwise a generic pathology prompt. Use '' to "
            "test an empty prompt."
        ),
    )
    parser.add_argument(
        "--target-prompt",
        default=None,
        help=(
            "Optional denoising prompt for official RF-Edit style debugging. "
            "Defaults to --source-prompt for pure reconstruction."
        ),
    )
    parser.add_argument(
        "--cross-image",
        action="store_true",
        help=(
            "Run STEP 3/4 cross-image RF-Solver experiment: invert target and "
            "reference images, then denoise from target noise while injecting "
            "reference single-block attention K/V."
        ),
    )
    parser.add_argument(
        "--cross-image-mode",
        choices=("v-only", "kv", "both"),
        default="kv",
        help=(
            "Cross-image attention ablation. v-only keeps target K and injects "
            "reference V; kv injects reference K and V; both runs both variants."
        ),
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help=(
            "Reference RGB image for --cross-image. Defaults to metadata "
            "reference_image when available."
        ),
    )
    parser.add_argument(
        "--reference-sample-id",
        default=None,
        help=(
            "Optional metadata sample_id to use as the cross-image reference. "
            "Use this when the default paired reference has nearly identical texture."
        ),
    )
    parser.add_argument(
        "--reference-metadata-index",
        type=int,
        default=None,
        help="Optional metadata index to use as the cross-image reference.",
    )
    parser.add_argument(
        "--reference-record-image-field",
        choices=("target_image", "reference_image"),
        default="target_image",
        help=(
            "Image field to read from --reference-sample-id/--reference-metadata-index/"
            "--auto-reference-by-texture. Defaults to target_image so another record "
            "becomes the appearance reference directly."
        ),
    )
    parser.add_argument(
        "--auto-reference-by-texture",
        action="store_true",
        help=(
            "Select a metadata reference automatically by maximizing simple RGB/"
            "gradient texture distance from the target image."
        ),
    )
    parser.add_argument(
        "--auto-reference-max-candidates",
        type=int,
        default=300,
        help="Maximum metadata records to scan for --auto-reference-by-texture.",
    )
    parser.add_argument(
        "--auto-reference-rank",
        type=int,
        default=0,
        help=(
            "Rank among texture-different candidates for --auto-reference-by-texture. "
            "0 selects the most different reference, 1 the second-most different, etc."
        ),
    )
    parser.add_argument(
        "--auto-reference-same-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the same dataset when auto-selecting a texture-different reference.",
    )
    parser.add_argument(
        "--auto-reference-different-case",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer a different case_id when auto-selecting a texture-different reference.",
    )
    parser.add_argument(
        "--reference-prompt",
        default=None,
        help=(
            "Prompt used for reference inversion. Defaults to the selected "
            "metadata prompt or --source-prompt."
        ),
    )
    parser.add_argument(
        "--cross-image-strength",
        type=float,
        default=1.0,
        help=(
            "Blend strength for injected reference K/V on image tokens. 1.0 is "
            "full replacement, 0.0 disables cross-image replacement."
        ),
    )
    parser.add_argument(
        "--inject-after-t",
        type=float,
        default=1.0,
        help=(
            "Only inject reference K/V when the denoise timestep t <= this value. "
            "FLUX t goes 1.0 (pure noise) -> 0.0 (clean image), so a smaller value "
            "injects later. 1.0 keeps the old behavior."
        ),
    )
    parser.add_argument(
        "--regional-mode",
        choices=("none", "tissue", "nuclei", "tissue_nuclei"),
        default="none",
        help=(
            "STEP 4 token-level regional attention mask. Applies only in "
            "--cross-image with image-token K/V injection."
        ),
    )
    parser.add_argument(
        "--target-tissue-mask",
        type=Path,
        default=None,
        help="Target tissue mask for --regional-mode. Defaults to metadata target_tissue_mask.",
    )
    parser.add_argument(
        "--reference-tissue-mask",
        type=Path,
        default=None,
        help="Reference tissue mask for --regional-mode. Defaults to metadata reference_tissue_mask.",
    )
    parser.add_argument(
        "--target-nuclei-mask",
        type=Path,
        default=None,
        help="Target nuclei mask for --regional-mode. Defaults to metadata target_nuclei_mask.",
    )
    parser.add_argument(
        "--reference-nuclei-mask",
        type=Path,
        default=None,
        help="Reference nuclei mask for --regional-mode. Defaults to metadata reference_nuclei_mask.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="Number of RF-Solver inversion and denoising steps.",
    )
    parser.add_argument(
        "--with-second-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RF-Solver's second-order update. Pass --no-with-second-order for first-order Euler.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Guidance scale for denoising. Inversion always uses guidance=1.0.",
    )
    parser.add_argument(
        "--name",
        default="flux-dev",
        help="RF-Solver-Edit model name. Default maps to black-forest-labs/FLUX.1-dev.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for FLUX inference, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Keep large components on CPU when possible to reduce GPU memory use.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit non-zero when both PSNR and SSIM miss the acceptance thresholds.",
    )
    parser.add_argument(
        "--vae-roundtrip-only",
        action="store_true",
        help=(
            "Diagnostic mode: only encode/decode with the FLUX autoencoder, "
            "then save artifacts and exit before loading T5/CLIP/flow model."
        ),
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=DEFAULT_PSNR_THRESHOLD,
        help="PSNR threshold for a green-light gate.",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=DEFAULT_SSIM_THRESHOLD,
        help="SSIM threshold for a green-light gate.",
    )
    parser.add_argument(
        "--rf-solver-root",
        type=Path,
        default=Path(os.environ["RF_SOLVER_ROOT"])
        if os.environ.get("RF_SOLVER_ROOT")
        else None,
        help=(
            "Path to RF-Solver-Edit/FLUX_Image_Edit. If set, its src/ directory "
            "is prepended to PYTHONPATH before importing flux modules."
        ),
    )
    parser.add_argument(
        "--flux-diffusers-root",
        type=Path,
        default=Path(os.environ["FLUX_DIFFUSERS_ROOT"])
        if os.environ.get("FLUX_DIFFUSERS_ROOT")
        else None,
        help=(
            "Optional local diffusers-format FLUX.1-dev directory. If omitted, "
            "the parent directory of FLUX_DEV is used when it contains "
            "text_encoder_2/tokenizer_2 and text_encoder/tokenizer."
        ),
    )
    parser.add_argument(
        "--t5-model-path",
        type=Path,
        default=Path(os.environ["T5_MODEL_PATH"])
        if os.environ.get("T5_MODEL_PATH")
        else None,
        help="Local T5 encoder model directory. Defaults to FLUX root/text_encoder_2 when present.",
    )
    parser.add_argument(
        "--t5-tokenizer-path",
        type=Path,
        default=Path(os.environ["T5_TOKENIZER_PATH"])
        if os.environ.get("T5_TOKENIZER_PATH")
        else None,
        help="Local T5 tokenizer directory. Defaults to FLUX root/tokenizer_2 when present.",
    )
    parser.add_argument(
        "--clip-model-path",
        type=Path,
        default=Path(os.environ["CLIP_MODEL_PATH"])
        if os.environ.get("CLIP_MODEL_PATH")
        else None,
        help="Local CLIP text encoder directory. Defaults to FLUX root/text_encoder when present.",
    )
    parser.add_argument(
        "--clip-tokenizer-path",
        type=Path,
        default=Path(os.environ["CLIP_TOKENIZER_PATH"])
        if os.environ.get("CLIP_TOKENIZER_PATH")
        else None,
        help="Local CLIP tokenizer directory. Defaults to FLUX root/tokenizer when present.",
    )
    parser.add_argument(
        "--allow-text-encoder-download",
        action="store_true",
        help=(
            "Allow RF-Solver-Edit to download google/t5-v1_1-xxl and "
            "openai/clip-vit-large-patch14 if local text encoder paths are not found. "
            "Default is false to avoid accidental large downloads."
        ),
    )
    parser.add_argument(
        "--inject-steps",
        type=int,
        default=0,
        help=(
            "Optional RF-Solver-Edit feature-injection steps. Default 0 keeps "
            "the gate as pure image->noise->image reconstruction."
        ),
    )
    parser.add_argument(
        "--cross-after-layer",
        type=int,
        default=DEFAULT_CROSS_AFTER_LAYER,
        help=(
            "First single transformer block index eligible for cross-image "
            "injection. Default 20 matches RF-Solver-Edit's single_blocks 20-37 setup."
        ),
    )
    parser.add_argument(
        "--metric-self-test",
        action="store_true",
        help="Run a tiny metrics sanity check without importing FLUX or loading models.",
    )
    parser.add_argument(
        "--metadata-self-test",
        action="store_true",
        help="Run a tiny metadata selection sanity check without importing FLUX or loading models.",
    )
    parser.add_argument(
        "--debug-features",
        action="store_true",
        help=(
            "Log actual RF-Edit feature save/load events from single stream blocks "
            "without editing the upstream RF-Solver-Edit checkout."
        ),
    )
    parser.add_argument(
        "--debug-max-events",
        type=int,
        default=12,
        help="Maximum number of detailed feature events to print per phase.",
    )
    parser.add_argument(
        "--save-feature-debug",
        action="store_true",
        help=(
            "Save cross-image feature-bank and injection summaries. Enabled "
            "automatically by --debug-features."
        ),
    )
    parser.add_argument(
        "--cross-self-test",
        action="store_true",
        help=(
            "Run lightweight K/V naming, token split, mask mapping, and regional "
            "fallback tests without importing FLUX or loading models."
        ),
    )
    return parser.parse_args()


def prepend_rf_solver_src(rf_solver_root: Path | None) -> None:
    if rf_solver_root is None:
        return
    candidates = [rf_solver_root / "src", rf_solver_root]
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate.resolve())
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def import_rf_solver_modules() -> dict[str, Any]:
    try:
        from flux.modules.layers import SingleStreamBlock
        from flux.sampling import denoise, get_schedule, prepare, unpack
        from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import RF-Solver-Edit's flux modules. Set "
            "RF_SOLVER_ROOT=/path/to/RF-Solver-Edit/FLUX_Image_Edit or add "
            "FLUX_Image_Edit/src to PYTHONPATH."
        ) from exc

    return {
        "denoise": denoise,
        "get_schedule": get_schedule,
        "prepare": prepare,
        "unpack": unpack,
        "configs": configs,
        "SingleStreamBlock": SingleStreamBlock,
        "load_ae": load_ae,
        "load_clip": load_clip,
        "load_flow_model": load_flow_model,
        "load_t5": load_t5,
    }


def read_metadata_records(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "pairs" in payload:
            records = payload["pairs"]
        elif "records" in payload:
            records = payload["records"]
        else:
            raise ValueError(
                f"Unsupported metadata object in {metadata_path}; expected key 'pairs' or 'records'."
            )
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(
            f"Unsupported metadata payload in {metadata_path}; expected list or object."
        )
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Metadata records must be JSON objects: {metadata_path}")
    return records


def select_metadata_record(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.metadata is None:
        return None
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {args.metadata}")
    records = read_metadata_records(args.metadata)
    if not records:
        raise ValueError(f"Metadata file is empty: {args.metadata}")
    if args.sample_id is not None:
        for record in records:
            if str(record.get("sample_id")) == args.sample_id:
                return record
        raise ValueError(f"sample_id {args.sample_id!r} not found in {args.metadata}")
    if args.metadata_index < 0 or args.metadata_index >= len(records):
        raise IndexError(
            f"--metadata-index {args.metadata_index} is out of range for "
            f"{len(records)} records in {args.metadata}"
        )
    return records[args.metadata_index]


def select_metadata_record_by_id_or_index(
    metadata_path: Path,
    *,
    sample_id: str | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    records = read_metadata_records(metadata_path)
    if sample_id is not None:
        for record in records:
            if str(record.get("sample_id")) == sample_id:
                return record
        raise ValueError(f"sample_id {sample_id!r} not found in {metadata_path}")
    if index is None:
        raise ValueError("sample_id or index is required.")
    if index < 0 or index >= len(records):
        raise IndexError(
            f"reference metadata index {index} is out of range for "
            f"{len(records)} records in {metadata_path}"
        )
    return records[index]


def texture_descriptor_from_image(path: Path, size: int = 128) -> np.ndarray:
    image = load_rgb_image(path).resize((size, size), Image.Resampling.BICUBIC)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    center = gray[1:-1, 1:-1]
    lap = np.abs(
        center * 4.0
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )

    features: list[np.ndarray] = [
        rgb.mean(axis=(0, 1)) * 2.0,
        rgb.std(axis=(0, 1)) * 2.0,
        np.percentile(rgb.reshape(-1, 3), [10, 50, 90], axis=0).reshape(-1),
        np.asarray(
            [
                grad.mean(),
                grad.std(),
                np.percentile(grad, 90),
                lap.mean(),
                lap.std(),
                np.percentile(lap, 90),
            ],
            dtype=np.float32,
        )
        * 4.0,
    ]
    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist_features.append(hist / max(1.0, float(hist.sum())))
    features.append(np.concatenate(hist_features, axis=0) * 0.5)
    return np.concatenate([feature.astype(np.float32).reshape(-1) for feature in features])


def subsample_records_evenly(
    records: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    if max_candidates <= 0 or len(records) <= max_candidates:
        return records
    indices = np.linspace(0, len(records) - 1, num=max_candidates, dtype=int)
    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for index in indices.tolist():
        if index in seen:
            continue
        seen.add(index)
        selected.append(records[index])
    return selected


def select_texture_different_reference_record(
    *,
    metadata_path: Path,
    target_record: dict[str, Any] | None,
    target_image_path: Path,
    image_field: str,
    max_candidates: int,
    rank: int,
    same_dataset: bool,
    different_case: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if rank < 0:
        raise ValueError("--auto-reference-rank must be >= 0.")
    records = read_metadata_records(metadata_path)
    target_sample_id = str(target_record.get("sample_id")) if target_record else None
    target_case_id = str(target_record.get("case_id")) if target_record and target_record.get("case_id") is not None else None
    target_dataset = str(target_record.get("dataset")) if target_record and target_record.get("dataset") is not None else None

    candidates = [
        record
        for record in records
        if record.get(image_field)
        and str(record.get("sample_id")) != target_sample_id
    ]
    if same_dataset and target_dataset is not None:
        same_dataset_candidates = [
            record for record in candidates if str(record.get("dataset")) == target_dataset
        ]
        if same_dataset_candidates:
            candidates = same_dataset_candidates
    if different_case and target_case_id is not None:
        different_case_candidates = [
            record for record in candidates if str(record.get("case_id")) != target_case_id
        ]
        if different_case_candidates:
            candidates = different_case_candidates
    candidates = subsample_records_evenly(candidates, max_candidates)
    if not candidates:
        raise ValueError(
            f"No metadata candidates with image field {image_field!r} for auto reference selection."
        )

    target_descriptor = texture_descriptor_from_image(target_image_path)
    scored_candidates: list[tuple[float, dict[str, Any]]] = []
    scored = 0
    skipped: list[str] = []
    for candidate in candidates:
        candidate_path = Path(str(candidate[image_field]))
        try:
            descriptor = texture_descriptor_from_image(candidate_path)
        except Exception as exc:  # pragma: no cover - defensive for server-side missing files
            skipped.append(f"{candidate.get('sample_id')}: {exc}")
            continue
        score = float(np.linalg.norm(target_descriptor - descriptor))
        scored += 1
        scored_candidates.append((score, candidate))

    if not scored_candidates:
        raise ValueError(
            "Could not score any auto-reference candidates. "
            f"Skipped examples: {skipped[:5]}"
        )
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    selected_index = min(rank, len(scored_candidates) - 1)
    best_score, best_record = scored_candidates[selected_index]
    summary = {
        "mode": "auto_texture",
        "image_field": image_field,
        "requested_rank": rank,
        "selected_rank": selected_index,
        "target_sample_id": target_sample_id,
        "target_case_id": target_case_id,
        "target_dataset": target_dataset,
        "selected_sample_id": best_record.get("sample_id"),
        "selected_case_id": best_record.get("case_id"),
        "selected_dataset": best_record.get("dataset"),
        "selected_image": best_record.get(image_field),
        "texture_distance": best_score,
        "candidate_count": len(candidates),
        "scored_count": scored,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:5],
        "top_candidates": [
            {
                "rank": index,
                "texture_distance": score,
                "sample_id": record.get("sample_id"),
                "case_id": record.get("case_id"),
                "dataset": record.get("dataset"),
                "image": record.get(image_field),
            }
            for index, (score, record) in enumerate(scored_candidates[:10])
        ],
        "same_dataset": same_dataset,
        "different_case": different_case,
    }
    return best_record, summary


def resolve_run_inputs(args: argparse.Namespace) -> dict[str, Any]:
    record = select_metadata_record(args)
    reference_record: dict[str, Any] | None = None
    image_path = args.image
    reference_image_path = getattr(args, "reference_image", None)
    target_tissue_mask = getattr(args, "target_tissue_mask", None)
    reference_tissue_mask = getattr(args, "reference_tissue_mask", None)
    target_nuclei_mask = getattr(args, "target_nuclei_mask", None)
    reference_nuclei_mask = getattr(args, "reference_nuclei_mask", None)
    source_prompt = args.source_prompt
    reference_prompt = getattr(args, "reference_prompt", None)
    prompt_source = "cli" if source_prompt is not None else "default"
    reference_prompt_source = "cli" if reference_prompt is not None else "source_prompt"
    reference_selection: dict[str, Any] | None = None
    manual_reference_image = getattr(args, "reference_image", None) is not None

    if record is not None:
        if image_path is None:
            image_value = record.get(args.image_field)
            if not image_value:
                raise KeyError(
                    f"Selected metadata record has no {args.image_field!r} field."
                )
            image_path = Path(str(image_value))
        if source_prompt is None:
            source_prompt = str(record.get(args.prompt_field, ""))
            prompt_source = f"metadata.{args.prompt_field}"
        if reference_image_path is None and record.get("reference_image"):
            reference_image_path = Path(str(record["reference_image"]))
        if reference_prompt is None and record.get(args.prompt_field) is not None:
            reference_prompt = str(record.get(args.prompt_field, ""))
            reference_prompt_source = f"metadata.{args.prompt_field}"
        if target_tissue_mask is None and record.get("target_tissue_mask"):
            target_tissue_mask = Path(str(record["target_tissue_mask"]))
        if (
            not manual_reference_image
            and reference_tissue_mask is None
            and record.get("reference_tissue_mask")
        ):
            reference_tissue_mask = Path(str(record["reference_tissue_mask"]))
        if target_nuclei_mask is None and record.get("target_nuclei_mask"):
            target_nuclei_mask = Path(str(record["target_nuclei_mask"]))
        if (
            not manual_reference_image
            and reference_nuclei_mask is None
            and record.get("reference_nuclei_mask")
        ):
            reference_nuclei_mask = Path(str(record["reference_nuclei_mask"]))

    if image_path is None:
        raise ValueError("--image is required unless --metadata supplies an image.")

    should_override_reference = args.metadata is not None and (
        getattr(args, "reference_sample_id", None) is not None
        or getattr(args, "reference_metadata_index", None) is not None
        or getattr(args, "auto_reference_by_texture", False)
    )
    if should_override_reference:
        if getattr(args, "auto_reference_by_texture", False):
            reference_record, reference_selection = select_texture_different_reference_record(
                metadata_path=args.metadata,
                target_record=record,
                target_image_path=image_path,
                image_field=args.reference_record_image_field,
                max_candidates=args.auto_reference_max_candidates,
                rank=args.auto_reference_rank,
                same_dataset=args.auto_reference_same_dataset,
                different_case=args.auto_reference_different_case,
            )
        else:
            reference_record = select_metadata_record_by_id_or_index(
                args.metadata,
                sample_id=args.reference_sample_id,
                index=args.reference_metadata_index,
            )
            reference_selection = {
                "mode": "manual_metadata_reference",
                "image_field": args.reference_record_image_field,
                "selected_sample_id": reference_record.get("sample_id"),
                "selected_case_id": reference_record.get("case_id"),
                "selected_dataset": reference_record.get("dataset"),
            }
        field = args.reference_record_image_field
        image_value = reference_record.get(field)
        if not image_value:
            raise KeyError(f"Selected reference metadata record has no {field!r} field.")
        reference_image_path = Path(str(image_value))
        if getattr(args, "reference_prompt", None) is None and reference_record.get(args.prompt_field) is not None:
            reference_prompt = str(reference_record.get(args.prompt_field, ""))
            reference_prompt_source = f"reference_metadata.{args.prompt_field}"
        if field == "target_image" and reference_record.get("target_tissue_mask"):
            reference_tissue_mask = Path(str(reference_record["target_tissue_mask"]))
        if field == "target_image" and reference_record.get("target_nuclei_mask"):
            reference_nuclei_mask = Path(str(reference_record["target_nuclei_mask"]))
        if field == "reference_image" and reference_record.get("reference_tissue_mask"):
            reference_tissue_mask = Path(str(reference_record["reference_tissue_mask"]))
        if field == "reference_image" and reference_record.get("reference_nuclei_mask"):
            reference_nuclei_mask = Path(str(reference_record["reference_nuclei_mask"]))

    if source_prompt is None:
        source_prompt = DEFAULT_SOURCE_PROMPT
    if reference_prompt is None:
        reference_prompt = source_prompt

    output_dir = args.output_dir
    if output_dir is None:
        if record is None:
            root = (
                "phase5_runs/rf_solver_flux_cross"
                if getattr(args, "cross_image", False)
                else "phase5_runs/rf_solver_flux_reconstruct"
            )
            output_dir = Path(root) / "pathology_25"
        else:
            sample_id = str(record.get("sample_id") or Path(image_path).stem)
            safe_sample_id = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in sample_id
            )
            suffix = "target" if args.image_field == "target_image" else "reference"
            root = (
                "phase5_runs/rf_solver_flux_cross"
                if getattr(args, "cross_image", False)
                else "phase5_runs/rf_solver_flux_reconstruct"
            )
            output_dir = (
                Path(root)
                / f"{safe_sample_id}_{suffix}_{args.num_inference_steps}"
            )

    return {
        "record": record,
        "reference_record": reference_record,
        "reference_selection": reference_selection,
        "image_path": image_path,
        "reference_image_path": reference_image_path,
        "target_tissue_mask": target_tissue_mask,
        "reference_tissue_mask": reference_tissue_mask,
        "target_nuclei_mask": target_nuclei_mask,
        "reference_nuclei_mask": reference_nuclei_mask,
        "source_prompt": source_prompt,
        "reference_prompt": reference_prompt,
        "prompt_source": prompt_source,
        "reference_prompt_source": reference_prompt_source,
        "output_dir": output_dir,
    }


def metadata_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    keys = [
        "dataset",
        "sample_id",
        "reference_sample_id",
        "case_id",
        "pair_difficulty",
        "distance",
        "tissue_coverage_ratio",
        "area_coverage_ratio",
        "covered_target_tissue_ids",
        "missing_target_tissue_ids",
        "target_image",
        "reference_image",
        "target_tissue_mask",
        "reference_tissue_mask",
        "target_nuclei_mask",
        "reference_nuclei_mask",
        "prompt",
    ]
    return {key: record[key] for key in keys if key in record}


def infer_flux_diffusers_root(args: argparse.Namespace) -> Path | None:
    if args.flux_diffusers_root is not None:
        return args.flux_diffusers_root
    flux_dev = os.environ.get("FLUX_DEV")
    if not flux_dev:
        return None
    candidate = Path(flux_dev).expanduser().resolve().parent
    expected = [
        candidate / "text_encoder_2",
        candidate / "tokenizer_2",
        candidate / "text_encoder",
        candidate / "tokenizer",
    ]
    if all(path.exists() for path in expected):
        return candidate
    return None


def resolve_text_encoder_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    flux_root = infer_flux_diffusers_root(args)
    return {
        "flux_diffusers_root": flux_root,
        "t5_model_path": args.t5_model_path
        or (flux_root / "text_encoder_2" if flux_root is not None else None),
        "t5_tokenizer_path": args.t5_tokenizer_path
        or (flux_root / "tokenizer_2" if flux_root is not None else None),
        "clip_model_path": args.clip_model_path
        or (flux_root / "text_encoder" if flux_root is not None else None),
        "clip_tokenizer_path": args.clip_tokenizer_path
        or (flux_root / "tokenizer" if flux_root is not None else None),
    }


def path_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def validate_text_encoder_paths(paths: dict[str, Path | None]) -> bool:
    required = [
        paths["t5_model_path"],
        paths["t5_tokenizer_path"],
        paths["clip_model_path"],
        paths["clip_tokenizer_path"],
    ]
    return all(path_exists(path) for path in required)


def configure_local_flux_weight_paths(rf: dict[str, Any], args: argparse.Namespace) -> dict[str, str | None]:
    """Point RF-Solver-Edit configs at local safetensors before load_ae/load_flow_model.

    RF-Solver-Edit reads FLUX_DEV/AE when flux.util is imported. If the user only
    provides a local diffusers root, the upstream config fields stay None and its
    loader tries to download the gated FLUX repo. Fill those fields explicitly.
    """
    config = rf["configs"][args.name]
    flux_root = infer_flux_diffusers_root(args)
    flow_env_name = "FLUX_SCHNELL" if args.name == "flux-schnell" else "FLUX_DEV"
    flow_path = os.environ.get(flow_env_name)
    ae_path = os.environ.get("AE")

    if flow_path is None and flux_root is not None:
        flow_name = "flux1-schnell.safetensors" if args.name == "flux-schnell" else "flux1-dev.safetensors"
        candidate = flux_root / flow_name
        if candidate.exists():
            flow_path = str(candidate)
    if ae_path is None and flux_root is not None:
        candidate = flux_root / "ae.safetensors"
        if candidate.exists():
            ae_path = str(candidate)

    if flow_path is not None:
        config.ckpt_path = flow_path
    if ae_path is not None:
        config.ae_path = ae_path

    return {
        flow_env_name: config.ckpt_path,
        "AE": config.ae_path,
        "flux_diffusers_root": str(flux_root) if flux_root is not None else None,
    }


class LocalHFEmbedder(nn.Module):
    def __init__(
        self,
        *,
        tokenizer: Any,
        hf_module: nn.Module,
        max_length: int,
        is_clip: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.hf_module = hf_module.eval().requires_grad_(False)
        self.max_length = max_length
        self.output_key = "pooler_output" if is_clip else "last_hidden_state"

    def forward(self, text: list[str]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


def load_local_text_encoders(
    *,
    device: torch.device,
    max_length: int,
    paths: dict[str, Path | None],
) -> tuple[Any, Any]:
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

    print(f"Loading local T5 encoder: {paths['t5_model_path']}")
    print(f"Loading local T5 tokenizer: {paths['t5_tokenizer_path']}")
    t5 = LocalHFEmbedder(
        tokenizer=T5Tokenizer.from_pretrained(str(paths["t5_tokenizer_path"])),
        hf_module=T5EncoderModel.from_pretrained(
            str(paths["t5_model_path"]),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device),
        max_length=max_length,
        is_clip=False,
    )

    print(f"Loading local CLIP encoder: {paths['clip_model_path']}")
    print(f"Loading local CLIP tokenizer: {paths['clip_tokenizer_path']}")
    clip = LocalHFEmbedder(
        tokenizer=CLIPTokenizer.from_pretrained(str(paths["clip_tokenizer_path"])),
        hf_module=CLIPTextModel.from_pretrained(
            str(paths["clip_model_path"]),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device),
        max_length=77,
        is_clip=True,
    )
    return t5, clip


def validate_args(args: argparse.Namespace) -> None:
    if args.metric_self_test or args.metadata_self_test or args.cross_self_test:
        return
    resolved = resolve_run_inputs(args)
    if not resolved["image_path"].exists():
        raise FileNotFoundError(f"Input image does not exist: {resolved['image_path']}")
    if args.cross_image:
        reference_image_path = resolved["reference_image_path"]
        if reference_image_path is None:
            raise ValueError(
                "--cross-image requires --reference-image or metadata reference_image."
            )
        if not reference_image_path.exists():
            raise FileNotFoundError(f"Reference image does not exist: {reference_image_path}")
        if args.inject_steps < 0:
            raise ValueError("--inject-steps must be >= 0.")
        if args.cross_after_layer < 0:
            raise ValueError("--cross-after-layer must be >= 0.")
        if not (0.0 <= args.cross_image_strength <= 1.0):
            raise ValueError("--cross-image-strength must be in [0, 1].")
        if args.regional_mode != "none":
            required_masks: list[tuple[str, Path | None]] = []
            if args.regional_mode in {"tissue", "tissue_nuclei"}:
                required_masks.extend(
                    [
                        ("target_tissue_mask", resolved["target_tissue_mask"]),
                        ("reference_tissue_mask", resolved["reference_tissue_mask"]),
                    ]
                )
            if args.regional_mode in {"nuclei", "tissue_nuclei"}:
                required_masks.extend(
                    [
                        ("target_nuclei_mask", resolved["target_nuclei_mask"]),
                        ("reference_nuclei_mask", resolved["reference_nuclei_mask"]),
                    ]
                )
            for name, path in required_masks:
                if path is None:
                    raise ValueError(f"--regional-mode {args.regional_mode} requires {name}.")
                if not path.exists():
                    raise FileNotFoundError(f"{name} does not exist: {path}")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")
    if args.inject_steps < 0:
        raise ValueError("--inject-steps must be >= 0.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested --device {args.device}, but torch.cuda.is_available() is false."
        )
    if args.inject_steps > 0 and not args.device.startswith("cuda"):
        raise ValueError(
            "--inject-steps > 0 requires a CUDA device because RF-Solver-Edit's "
            "feature replay path moves stored tensors with .cuda()."
        )


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def crop_to_multiple(image: Image.Image, multiple: int = 16) -> tuple[Image.Image, dict[str, int]]:
    width, height = image.size
    cropped_width = width - (width % multiple)
    cropped_height = height - (height % multiple)
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError(
            f"Image is too small after cropping to a multiple of {multiple}: "
            f"{width}x{height}"
        )
    cropped = image.crop((0, 0, cropped_width, cropped_height))
    crop_info = {
        "original_width": width,
        "original_height": height,
        "cropped_width": cropped_width,
        "cropped_height": cropped_height,
        "crop_left": 0,
        "crop_top": 0,
    }
    return cropped, crop_info


@torch.inference_mode()
def encode_image(image: Image.Image, device: torch.device, ae: torch.nn.Module) -> torch.Tensor:
    image_np = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0
    tensor = tensor.unsqueeze(0).to(device)
    return ae.encode(tensor).to(torch.bfloat16)


@torch.inference_mode()
def decode_image(
    packed_latents: torch.Tensor,
    height: int,
    width: int,
    ae: torch.nn.Module,
    unpack: Callable[..., torch.Tensor],
    device: torch.device,
) -> Image.Image:
    latents = unpack(packed_latents.float(), height, width)
    autocast_context: contextlib.AbstractContextManager[Any]
    if device.type == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = contextlib.nullcontext()

    with autocast_context:
        decoded = ae.decode(latents)

    decoded = decoded.clamp(-1, 1)
    decoded = decoded[0].permute(1, 2, 0)
    array = (127.5 * (decoded + 1.0)).detach().cpu().byte().numpy()
    return Image.fromarray(array, mode="RGB")


@torch.inference_mode()
def decode_latents(
    latents: torch.Tensor,
    ae: torch.nn.Module,
    device: torch.device,
) -> Image.Image:
    autocast_context: contextlib.AbstractContextManager[Any]
    if device.type == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = contextlib.nullcontext()

    with autocast_context:
        decoded = ae.decode(latents.float())

    decoded = decoded.clamp(-1, 1)
    decoded = decoded[0].permute(1, 2, 0)
    array = (127.5 * (decoded + 1.0)).detach().cpu().byte().numpy()
    return Image.fromarray(array, mode="RGB")


@torch.inference_mode()
def denoise_first_order(
    model: torch.nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    inverse: bool,
    info: dict[str, Any],
    guidance: float = 4.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    inject_step = int(info.get("inject_step", 0))
    inject_list = [True] * inject_step + [False] * (
        len(timesteps[:-1]) - inject_step
    )
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )
        info["t"] = t_prev if inverse else t_curr
        info["inverse"] = inverse
        info["second_order"] = False
        info["inject"] = inject_list[i]
        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
        )
        img = img + (t_prev - t_curr) * pred
    return img, info


def image_metrics(original: Image.Image, reconstruction: Image.Image) -> dict[str, Any]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    original_arr = np.asarray(original, dtype=np.uint8)
    reconstruction_arr = np.asarray(reconstruction, dtype=np.uint8)
    if original_arr.shape != reconstruction_arr.shape:
        height = min(original_arr.shape[0], reconstruction_arr.shape[0])
        width = min(original_arr.shape[1], reconstruction_arr.shape[1])
        original_arr = original_arr[:height, :width]
        reconstruction_arr = reconstruction_arr[:height, :width]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psnr = float(
            peak_signal_noise_ratio(original_arr, reconstruction_arr, data_range=255)
        )
    ssim = float(
        structural_similarity(
            original_arr,
            reconstruction_arr,
            channel_axis=2,
            data_range=255,
        )
    )
    mean_abs_error = float(
        np.mean(
            np.abs(
                original_arr.astype(np.int16) - reconstruction_arr.astype(np.int16)
            )
        )
    )
    max_abs_error = int(
        np.max(
            np.abs(
                original_arr.astype(np.int16) - reconstruction_arr.astype(np.int16)
            )
        )
    )
    return {
        "psnr": "inf" if math.isinf(psnr) else psnr,
        "psnr_is_infinite": math.isinf(psnr),
        "ssim": ssim,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "comparison_shape": list(original_arr.shape),
    }


def metric_value_for_threshold(value: Any) -> float:
    if value == "inf":
        return math.inf
    return float(value)


def make_diff_image(original: Image.Image, reconstruction: Image.Image) -> Image.Image:
    original_arr = np.asarray(original, dtype=np.uint8)
    reconstruction_arr = np.asarray(reconstruction, dtype=np.uint8)
    height = min(original_arr.shape[0], reconstruction_arr.shape[0])
    width = min(original_arr.shape[1], reconstruction_arr.shape[1])
    diff = np.abs(
        original_arr[:height, :width].astype(np.int16)
        - reconstruction_arr[:height, :width].astype(np.int16)
    ).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def make_comparison_image(
    original: Image.Image, reconstruction: Image.Image, diff: Image.Image
) -> Image.Image:
    width, height = original.size
    label_height = 24
    gap = 8
    canvas = Image.new(
        "RGB",
        (width * 3 + gap * 2, height + label_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    panels = [
        ("original_cropped", original),
        ("reconstruction", reconstruction),
        ("absolute_diff", diff),
    ]
    for index, (label, panel) in enumerate(panels):
        x = index * (width + gap)
        canvas.paste(panel.resize((width, height)), (x, label_height))
        draw.text((x + 4, 4), label, fill=(0, 0, 0))
    return canvas


def make_labeled_grid(
    panels: list[tuple[str, Image.Image]],
    *,
    columns: int | None = None,
    panel_size: tuple[int, int] | None = None,
) -> Image.Image:
    if not panels:
        raise ValueError("make_labeled_grid requires at least one panel.")
    if columns is None:
        columns = len(panels)
    columns = max(1, int(columns))
    rows = int(math.ceil(len(panels) / columns))
    label_height = 24
    gap = 8
    if panel_size is None:
        width, height = panels[0][1].size
    else:
        width, height = panel_size
    canvas = Image.new(
        "RGB",
        (
            columns * width + (columns - 1) * gap,
            rows * (height + label_height) + (rows - 1) * gap,
        ),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row = index // columns
        col = index % columns
        x = col * (width + gap)
        y = row * (height + label_height + gap)
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
        canvas.paste(image.resize((width, height)), (x, y + label_height))
    return canvas


def write_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cross_feature_key(info: dict[str, Any], block_id: int, kind: str) -> str:
    return (
        f"{info['t']}_{info['second_order']}_{block_id}_"
        f"{info.get('type', 'single')}_{kind}"
    )


def infer_token_split(total_tokens: int, image_tokens: int = DEFAULT_IMAGE_TOKEN_COUNT) -> tuple[int, int]:
    if total_tokens <= image_tokens:
        raise ValueError(
            f"Cannot split FLUX single-block sequence with total_tokens={total_tokens} "
            f"and image_tokens={image_tokens}."
        )
    text_tokens = total_tokens - image_tokens
    return text_tokens, image_tokens


def resize_mask_to_token_labels(mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")
    side = int(round(float(num_tokens) ** 0.5))
    if side * side != int(num_tokens):
        raise ValueError(f"expected a square spatial token grid, got num_tokens={num_tokens}")
    labels = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(side, side),
        mode="nearest",
    )[:, 0]
    return labels.to(dtype=torch.long).flatten(1)


def load_mask_tensor(path: Path, crop_size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        mask_image = ImageOps.exif_transpose(image).convert("L")
        if crop_size is not None:
            width, height = crop_size
            mask_image = mask_image.crop((0, 0, int(width), int(height)))
        array = np.asarray(mask_image, dtype=np.int64)
    return torch.from_numpy(array).unsqueeze(0)


def nonzero_labels(labels: torch.Tensor) -> torch.Tensor:
    return torch.where(labels > 0, labels, torch.full_like(labels, -1))


@dataclass
class RegionalTokenLabels:
    mode: str
    target_tissue: torch.Tensor | None = None
    reference_tissue: torch.Tensor | None = None
    target_nuclei: torch.Tensor | None = None
    reference_nuclei: torch.Tensor | None = None
    target_composite: torch.Tensor | None = None
    reference_composite: torch.Tensor | None = None

    def summary(self) -> dict[str, Any]:
        def _counts(tensor: torch.Tensor | None) -> dict[str, int] | None:
            if tensor is None:
                return None
            unique, counts = torch.unique(tensor.detach().cpu(), return_counts=True)
            return {str(int(label)): int(count) for label, count in zip(unique, counts)}

        return {
            "mode": self.mode,
            "target_tissue_counts": _counts(self.target_tissue),
            "reference_tissue_counts": _counts(self.reference_tissue),
            "target_nuclei_counts": _counts(self.target_nuclei),
            "reference_nuclei_counts": _counts(self.reference_nuclei),
            "target_composite_counts": _counts(self.target_composite),
            "reference_composite_counts": _counts(self.reference_composite),
        }


def build_regional_token_labels(
    *,
    mode: str,
    target_tissue_mask: Path | None,
    reference_tissue_mask: Path | None,
    target_nuclei_mask: Path | None,
    reference_nuclei_mask: Path | None,
    num_image_tokens: int = DEFAULT_IMAGE_TOKEN_COUNT,
    target_size: tuple[int, int] | None = None,
    reference_size: tuple[int, int] | None = None,
) -> RegionalTokenLabels | None:
    if mode == "none":
        return None
    if mode in {"tissue", "tissue_nuclei"} and (
        target_tissue_mask is None or reference_tissue_mask is None
    ):
        raise ValueError(
            f"--regional-mode {mode} requires target/reference tissue masks."
        )
    if mode in {"nuclei", "tissue_nuclei"} and (
        target_nuclei_mask is None or reference_nuclei_mask is None
    ):
        raise ValueError(
            f"--regional-mode {mode} requires target/reference nuclei masks."
        )

    target_tissue = None
    reference_tissue = None
    target_nuclei = None
    reference_nuclei = None
    target_composite = None
    reference_composite = None

    if target_tissue_mask is not None and reference_tissue_mask is not None:
        target_tissue = nonzero_labels(
            resize_mask_to_token_labels(
                load_mask_tensor(target_tissue_mask, target_size),
                num_image_tokens,
            )
        )
        reference_tissue = nonzero_labels(
            resize_mask_to_token_labels(
                load_mask_tensor(reference_tissue_mask, reference_size),
                num_image_tokens,
            )
        )
    if target_nuclei_mask is not None and reference_nuclei_mask is not None:
        target_nuclei = nonzero_labels(
            resize_mask_to_token_labels(
                load_mask_tensor(target_nuclei_mask, target_size),
                num_image_tokens,
            )
        )
        reference_nuclei = nonzero_labels(
            resize_mask_to_token_labels(
                load_mask_tensor(reference_nuclei_mask, reference_size),
                num_image_tokens,
            )
        )
    if mode == "tissue_nuclei":
        if target_tissue is None or reference_tissue is None or target_nuclei is None or reference_nuclei is None:
            raise ValueError("--regional-mode tissue_nuclei requires both tissue and nuclei masks.")
        target_composite = torch.where(
            target_tissue > 0,
            target_tissue * 256 + torch.clamp(target_nuclei, min=0),
            torch.full_like(target_tissue, -1),
        )
        reference_composite = torch.where(
            reference_tissue > 0,
            reference_tissue * 256 + torch.clamp(reference_nuclei, min=0),
            torch.full_like(reference_tissue, -1),
        )

    return RegionalTokenLabels(
        mode=mode,
        target_tissue=target_tissue,
        reference_tissue=reference_tissue,
        target_nuclei=target_nuclei,
        reference_nuclei=reference_nuclei,
        target_composite=target_composite,
        reference_composite=reference_composite,
    )


def token_label_grid_image(labels: torch.Tensor, scale: int = 16) -> Image.Image:
    flat = labels.detach().cpu().flatten()
    side = int(round(float(flat.numel()) ** 0.5))
    if side * side != flat.numel():
        raise ValueError(f"token labels must form a square grid, got {flat.numel()} labels")
    label_grid = flat.reshape(side, side).numpy().astype(np.int64)
    valid = label_grid[label_grid >= 0]
    if valid.size == 0:
        norm = np.zeros_like(label_grid, dtype=np.uint8)
    else:
        unique_values = {int(value): index + 1 for index, value in enumerate(sorted(set(valid.tolist())))}
        norm = np.zeros_like(label_grid, dtype=np.uint8)
        for value, mapped in unique_values.items():
            norm[label_grid == value] = int((mapped * 255) / max(1, len(unique_values)))
    image = Image.fromarray(norm, mode="L").convert("RGB")
    return image.resize((side * scale, side * scale), Image.Resampling.NEAREST)


def save_regional_token_overlays(labels: RegionalTokenLabels | None, output_dir: Path) -> dict[str, str]:
    if labels is None:
        return {}
    overlay_dir = output_dir / "token_masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for name, tensor in [
        ("target_tissue", labels.target_tissue),
        ("reference_tissue", labels.reference_tissue),
        ("target_nuclei", labels.target_nuclei),
        ("reference_nuclei", labels.reference_nuclei),
        ("target_composite", labels.target_composite),
        ("reference_composite", labels.reference_composite),
    ]:
        if tensor is None:
            continue
        path = overlay_dir / f"{name}_32x32.png"
        token_label_grid_image(tensor[0]).save(path)
        artifacts[name] = str(path)
    return artifacts


def build_regional_attention_mask(
    *,
    total_tokens: int,
    text_tokens: int,
    image_tokens: int,
    labels: RegionalTokenLabels,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_slice = slice(text_tokens, total_tokens)
    mask = torch.zeros((1, 1, total_tokens, total_tokens), dtype=torch.bool, device=device)
    mask[:, :, :text_tokens, :text_tokens] = True
    mask[:, :, target_slice, :text_tokens] = True

    def _labels_for_exact() -> tuple[torch.Tensor, torch.Tensor]:
        if labels.mode == "tissue":
            if labels.target_tissue is None or labels.reference_tissue is None:
                raise ValueError("tissue regional labels are missing.")
            return labels.target_tissue.to(device), labels.reference_tissue.to(device)
        if labels.mode == "nuclei":
            if labels.target_nuclei is None or labels.reference_nuclei is None:
                raise ValueError("nuclei regional labels are missing.")
            return labels.target_nuclei.to(device), labels.reference_nuclei.to(device)
        if labels.target_composite is None or labels.reference_composite is None:
            raise ValueError("tissue_nuclei composite labels are missing.")
        return labels.target_composite.to(device), labels.reference_composite.to(device)

    query_labels, key_labels = _labels_for_exact()
    fallback_tissue = labels.mode == "tissue_nuclei" and labels.target_tissue is not None and labels.reference_tissue is not None
    target_tissue = labels.target_tissue.to(device) if fallback_tissue else None
    reference_tissue = labels.reference_tissue.to(device) if fallback_tissue else None

    fallback_all_count = 0
    fallback_tissue_count = 0
    exact_count = 0
    for query_index in range(image_tokens):
        label = int(query_labels[0, query_index].item())
        allowed = key_labels[0] == label if label >= 0 else torch.zeros_like(key_labels[0], dtype=torch.bool)
        if bool(allowed.any()):
            exact_count += 1
        elif fallback_tissue and target_tissue is not None and reference_tissue is not None:
            tissue_label = int(target_tissue[0, query_index].item())
            allowed = reference_tissue[0] == tissue_label if tissue_label >= 0 else torch.zeros_like(reference_tissue[0], dtype=torch.bool)
            if bool(allowed.any()):
                fallback_tissue_count += 1
            else:
                allowed = torch.ones((image_tokens,), dtype=torch.bool, device=device)
                fallback_all_count += 1
        else:
            allowed = torch.ones((image_tokens,), dtype=torch.bool, device=device)
            fallback_all_count += 1
        mask[:, :, text_tokens + query_index, text_tokens:total_tokens] = allowed

    allowed_per_image_query = mask[0, 0, text_tokens:total_tokens].sum(dim=1)
    stats = {
        "mode": labels.mode,
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "exact_query_count": exact_count,
        "fallback_tissue_query_count": fallback_tissue_count,
        "fallback_all_query_count": fallback_all_count,
        "allowed_tokens_per_image_query_min": int(allowed_per_image_query.min().item()),
        "allowed_tokens_per_image_query_max": int(allowed_per_image_query.max().item()),
        "allowed_tokens_per_image_query_mean": float(allowed_per_image_query.float().mean().item()),
    }
    return mask, stats


@dataclass
class CrossImageFeatureBank:
    after_layer: int = DEFAULT_CROSS_AFTER_LAYER
    text_token_count: int = DEFAULT_TEXT_TOKEN_COUNT
    image_token_count: int = DEFAULT_IMAGE_TOKEN_COUNT
    strength: float = 1.0
    regional_labels: RegionalTokenLabels | None = None
    store_target_features: bool = True
    phase: str = "idle"
    denoise_mode: str = "kv"
    inject_after_t: float = 1.0
    features: dict[str, dict[str, dict[str, torch.Tensor]]] = field(
        default_factory=lambda: {"target": {}, "reference": {}}
    )
    events: list[dict[str, Any]] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    regional_stats: list[dict[str, Any]] = field(default_factory=list)

    def should_touch(self, info: dict[str, Any], block_id: int) -> bool:
        if not (bool(info.get("inject")) and block_id >= self.after_layer):
            return False
        if self.phase == "cross_denoise" and not bool(info.get("inverse")):
            t_curr = float(info.get("t", 0.0))
            if t_curr > self.inject_after_t:
                return False
        return True

    def store(self, role: str, info: dict[str, Any], block_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if role == "target" and not self.store_target_features:
            return
        key_base = cross_feature_key(info, block_id, "KV")
        self.features.setdefault(role, {})[key_base] = {
            "K": k.detach().to("cpu"),
            "V": v.detach().to("cpu"),
        }
        if len(self.events) < 2000:
            self.events.append(
                {
                    "phase": self.phase,
                    "role": role,
                    "action": "store_kv",
                    "key": key_base,
                    "block_id": block_id,
                    "t": float(info["t"]),
                    "second_order": bool(info["second_order"]),
                    "k_shape": list(k.shape),
                    "v_shape": list(v.shape),
                }
            )

    def get_reference(self, info: dict[str, Any], block_id: int) -> dict[str, torch.Tensor] | None:
        key_base = cross_feature_key(info, block_id, "KV")
        payload = self.features.get("reference", {}).get(key_base)
        if payload is None:
            self.missing_keys.append(key_base)
        return payload

    def inject(
        self,
        *,
        info: dict[str, Any],
        block_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        payload = self.get_reference(info, block_id)
        if payload is None:
            return k, v, None
        ref_k = payload["K"].to(device=k.device, dtype=k.dtype)
        ref_v = payload["V"].to(device=v.device, dtype=v.dtype)
        if ref_k.shape != k.shape or ref_v.shape != v.shape:
            self.missing_keys.append(
                f"shape_mismatch:{cross_feature_key(info, block_id, 'KV')}:"
                f"k={tuple(k.shape)} ref_k={tuple(ref_k.shape)} "
                f"v={tuple(v.shape)} ref_v={tuple(ref_v.shape)}"
            )
            return k, v, None

        total_tokens = int(k.shape[2])
        text_tokens, image_tokens = infer_token_split(total_tokens, self.image_token_count)
        self.text_token_count = text_tokens
        target_slice = slice(text_tokens, total_tokens)
        strength = float(self.strength)
        next_k = k.clone() if self.denoise_mode == "kv" and strength > 0 else k
        next_v = v.clone() if strength > 0 else v
        if strength >= 1.0:
            if self.denoise_mode == "kv":
                next_k[:, :, target_slice, :] = ref_k[:, :, target_slice, :]
            next_v[:, :, target_slice, :] = ref_v[:, :, target_slice, :]
        elif strength > 0:
            if self.denoise_mode == "kv":
                next_k[:, :, target_slice, :] = torch.lerp(
                    next_k[:, :, target_slice, :],
                    ref_k[:, :, target_slice, :],
                    strength,
                )
            next_v[:, :, target_slice, :] = torch.lerp(
                next_v[:, :, target_slice, :],
                ref_v[:, :, target_slice, :],
                strength,
            )

        attn_mask = None
        if self.regional_labels is not None:
            attn_mask, stats = build_regional_attention_mask(
                total_tokens=total_tokens,
                text_tokens=text_tokens,
                image_tokens=image_tokens,
                labels=self.regional_labels,
                device=k.device,
            )
            stats.update(
                {
                    "phase": self.phase,
                    "denoise_mode": self.denoise_mode,
                    "block_id": block_id,
                    "t": float(info["t"]),
                    "second_order": bool(info["second_order"]),
                }
            )
            if len(self.regional_stats) < 256:
                self.regional_stats.append(stats)

        if len(self.events) < 2000:
            self.events.append(
                {
                    "phase": self.phase,
                    "role": "reference",
                    "action": "inject_ref_kv" if self.denoise_mode == "kv" else "inject_ref_v",
                    "key": cross_feature_key(info, block_id, "KV"),
                    "block_id": block_id,
                    "t": float(info["t"]),
                    "second_order": bool(info["second_order"]),
                    "mode": self.denoise_mode,
                    "strength": strength,
                    "regional_mode": self.regional_labels.mode if self.regional_labels is not None else "none",
                    "tokens_total": total_tokens,
                    "text_tokens": text_tokens,
                    "image_tokens": image_tokens,
                }
            )
        return next_k, next_v, attn_mask

    def summary(self) -> dict[str, Any]:
        def _role_summary(role: str) -> dict[str, Any]:
            entries = self.features.get(role, {})
            shapes: dict[str, Any] = {}
            if entries:
                first_key = next(iter(entries))
                shapes = {
                    "example_key": first_key,
                    "example_k_shape": list(entries[first_key]["K"].shape),
                    "example_v_shape": list(entries[first_key]["V"].shape),
                }
            blocks = sorted(
                {
                    int(key.split("_")[2])
                    for key in entries
                    if len(key.split("_")) >= 3 and key.split("_")[2].isdigit()
                }
            )
            return {
                "feature_count": len(entries),
                "block_ids": blocks,
                **shapes,
            }

        return {
            "after_layer": self.after_layer,
            "text_token_count": self.text_token_count,
            "image_token_count": self.image_token_count,
            "strength": self.strength,
            "denoise_mode": self.denoise_mode,
            "inject_after_t": self.inject_after_t,
            "target": _role_summary("target"),
            "reference": _role_summary("reference"),
            "missing_key_count": len(self.missing_keys),
            "missing_keys_sample": self.missing_keys[:20],
            "event_count": len(self.events),
            "events_sample": self.events[:40],
            "regional_labels": self.regional_labels.summary() if self.regional_labels is not None else None,
            "regional_stats_sample": self.regional_stats[:20],
        }


def install_cross_image_forward_patch(single_stream_block_cls: type[Any]) -> None:
    if getattr(single_stream_block_cls, "_rf_cross_image_patched", False):
        return

    from einops import rearrange
    from flux.math import apply_rope, attention

    def masked_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pe: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k = apply_rope(q, k, pe)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return x

    def cross_forward(
        self: torch.nn.Module,
        x: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        info: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim],
            dim=-1,
        )
        q, k, v = rearrange(
            qkv,
            "B L (K H D) -> K B H L D",
            K=3,
            H=self.num_heads,
        )
        q, k = self.norm(q, k, v)

        bank: CrossImageFeatureBank | None = None
        if info is not None:
            bank = info.get("_cross_bank")
        attn_mask = None
        block_id = int(info.get("id", -1)) if info is not None else -1
        if bank is not None and info is not None and bank.should_touch(info, block_id):
            if info.get("inverse"):
                if bank.phase == "target_inversion":
                    bank.store("target", info, block_id, k, v)
                elif bank.phase == "reference_inversion":
                    bank.store("reference", info, block_id, k, v)
            elif bank.phase == "cross_denoise":
                k, v, attn_mask = bank.inject(
                    info=info,
                    block_id=block_id,
                    k=k,
                    v=v,
                )

        if attn_mask is None:
            attn = attention(q, k, v, pe=pe)
        else:
            attn = masked_attention(q, k, v, pe, attn_mask)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        return x + mod.gate * output, info

    single_stream_block_cls.forward = cross_forward
    single_stream_block_cls._rf_cross_image_patched = True


def install_feature_debug_hook(single_stream_block_cls: type[Any]) -> None:
    if getattr(single_stream_block_cls, "_rf_feature_debug_hooked", False):
        return

    original_forward = single_stream_block_cls.forward

    def debug_forward(self: torch.nn.Module, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor, info: dict[str, Any]):
        if info is not None and info.get("inject") and int(info.get("id", -1)) > 19:
            block_id = int(info["id"])
            num_heads = int(getattr(self, "num_heads"))
            hidden_size = int(getattr(self, "hidden_size"))
            event = {
                "phase": "inversion_save" if info.get("inverse") else "denoise_load",
                "t": float(info["t"]),
                "second_order": bool(info["second_order"]),
                "block_id": block_id,
                "block_name": f"single_transformer_blocks.{block_id}",
                "feature_name": (
                    f"{info['t']}_{info['second_order']}_{block_id}_"
                    f"{info.get('type', 'single')}_V"
                ),
                "x_shape": list(x.shape),
                "v_shape": [
                    int(x.shape[0]),
                    num_heads,
                    int(x.shape[1]),
                    hidden_size // num_heads,
                ],
                "tokens_total": int(x.shape[1]),
                "heads": num_heads,
                "head_dim": hidden_size // num_heads,
            }
            info.setdefault("_debug_events", []).append(event)
        return original_forward(self, x, vec, pe, info)

    single_stream_block_cls.forward = debug_forward
    single_stream_block_cls._rf_feature_debug_hooked = True


def summarize_debug_events(events: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    selected = [event for event in events if event["phase"] == phase]
    if not selected:
        return {
            "phase": phase,
            "event_count": 0,
            "block_ids": [],
            "step_t_values": [],
            "second_order_values": [],
            "example_v_shape": None,
        }
    block_ids = sorted({int(event["block_id"]) for event in selected})
    step_t_values = sorted({round(float(event["t"]), 10) for event in selected}, reverse=True)
    second_order_values = sorted({bool(event["second_order"]) for event in selected})
    return {
        "phase": phase,
        "event_count": len(selected),
        "block_ids": block_ids,
        "block_count": len(block_ids),
        "step_t_values": step_t_values,
        "step_count": len(step_t_values),
        "second_order_values": second_order_values,
        "example_feature_name": selected[0]["feature_name"],
        "example_x_shape": selected[0]["x_shape"],
        "example_v_shape": selected[0]["v_shape"],
        "tokens_total": selected[0]["tokens_total"],
        "heads": selected[0]["heads"],
        "head_dim": selected[0]["head_dim"],
    }


def print_debug_summary(
    events: list[dict[str, Any]],
    phase: str,
    max_events: int,
) -> dict[str, Any]:
    summary = summarize_debug_events(events, phase)
    print(f"[feature-debug] {phase}: {json.dumps(summary, sort_keys=True)}")
    for event in [event for event in events if event["phase"] == phase][:max_events]:
        print(f"[feature-debug] event: {json.dumps(event, sort_keys=True)}")
    return summary


def run_metric_self_test() -> int:
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    metrics = image_metrics(image, image)
    passed = metric_value_for_threshold(metrics["psnr"]) > DEFAULT_PSNR_THRESHOLD
    passed = passed and metrics["ssim"] == 1.0
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not passed:
        print("Metric self-test failed.", file=sys.stderr)
        return 1
    return 0


def run_metadata_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "target.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB").save(
            image_path
        )
        records = [
            {
                "dataset": "BCSS",
                "sample_id": "sample_a",
                "target_image": str(image_path),
                "reference_image": str(image_path),
                "prompt": "metadata prompt",
            }
        ]
        metadata_path = root / "metadata.json"
        metadata_path.write_text(json.dumps({"pairs": records}), encoding="utf-8")
        args = argparse.Namespace(
            metadata=metadata_path,
            metadata_index=0,
            sample_id="sample_a",
            image=None,
            image_field="target_image",
            prompt_field="prompt",
            source_prompt=None,
            output_dir=None,
            num_inference_steps=25,
        )
        resolved = resolve_run_inputs(args)
        passed = (
            resolved["image_path"] == image_path
            and resolved["source_prompt"] == "metadata prompt"
            and resolved["record"]["sample_id"] == "sample_a"
        )
        print(
            json.dumps(
                {
                    "image_path": str(resolved["image_path"]),
                    "source_prompt": resolved["source_prompt"],
                    "output_dir": str(resolved["output_dir"]),
                    "record": metadata_summary(resolved["record"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not passed:
            print("Metadata self-test failed.", file=sys.stderr)
            return 1
    return 0


def run_cross_self_test() -> int:
    info = {"t": 0.5, "second_order": True, "type": "single"}
    key = cross_feature_key(info, 20, "KV")
    token_split = infer_token_split(1536, 1024)

    mask = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 2, 2, 0],
                [3, 3, 2, 0],
                [3, 3, 2, 0],
            ]
        ],
        dtype=torch.long,
    )
    labels = resize_mask_to_token_labels(mask, 4)

    regional = RegionalTokenLabels(
        mode="tissue",
        target_tissue=torch.tensor([[9, 10, -1, 10]], dtype=torch.long),
        reference_tissue=torch.tensor([[1, 2, 3, -1]], dtype=torch.long),
    )
    attn_mask, fallback_stats = build_regional_attention_mask(
        total_tokens=6,
        text_tokens=2,
        image_tokens=4,
        labels=regional,
        device=torch.device("cpu"),
    )
    checks = {
        "feature_key": key,
        "feature_key_pass": key.endswith("_20_single_KV"),
        "token_split": token_split,
        "token_split_pass": token_split == (512, 1024),
        "mask_labels": labels.tolist(),
        "mask_mapping_pass": labels.tolist() == [[1, 0, 3, 2]],
        "fallback_stats": fallback_stats,
        "fallback_pass": fallback_stats["fallback_all_query_count"] == 4
        and bool(attn_mask[:, :, 2:, 2:].all()),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(
        bool(checks[name])
        for name in [
            "feature_key_pass",
            "token_split_pass",
            "mask_mapping_pass",
            "fallback_pass",
        ]
    ):
        print("Cross-image self-test failed.", file=sys.stderr)
        return 1
    return 0


def maybe_empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def ensure_same_packed_token_count(
    target_prepared: dict[str, Any],
    reference_prepared: dict[str, Any],
) -> int:
    target_tokens = int(target_prepared["img"].shape[1])
    reference_tokens = int(reference_prepared["img"].shape[1])
    if target_tokens != reference_tokens:
        raise ValueError(
            "Target/reference packed image token counts differ; cross-image K/V "
            f"replacement requires equal lengths. target={target_tokens}, "
            f"reference={reference_tokens}."
        )
    return target_tokens


def mode_file_stem(mode: str, regional_mode: str) -> str:
    clean_mode = mode.replace("-", "_")
    if regional_mode == "none":
        return f"cross_{clean_mode}"
    return f"cross_{clean_mode}_regional_{regional_mode}"


def run_cross_image_reconstruction(args: argparse.Namespace) -> int:
    resolved = resolve_run_inputs(args)
    image_path: Path = resolved["image_path"]
    reference_image_path: Path | None = resolved["reference_image_path"]
    if reference_image_path is None:
        raise ValueError("--cross-image requires a reference image.")
    output_dir: Path = resolved["output_dir"]
    source_prompt: str = resolved["source_prompt"]
    reference_prompt: str = resolved["reference_prompt"]
    target_prompt = source_prompt if args.target_prompt is None else args.target_prompt
    record: dict[str, Any] | None = resolved["record"]
    reference_record: dict[str, Any] | None = resolved["reference_record"]

    prepend_rf_solver_src(args.rf_solver_root)
    rf = import_rf_solver_modules()
    install_cross_image_forward_patch(rf["SingleStreamBlock"])
    if args.name not in rf["configs"]:
        available = ", ".join(sorted(rf["configs"].keys()))
        raise ValueError(f"Unknown --name {args.name!r}; available names: {available}")
    local_weight_paths = configure_local_flux_weight_paths(rf, args)

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch_device = torch.device(args.device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)
    started_at = time.perf_counter()

    target_original, target_crop_info = crop_to_multiple(load_rgb_image(image_path))
    reference_original, reference_crop_info = crop_to_multiple(load_rgb_image(reference_image_path))
    if target_original.size != reference_original.size:
        raise ValueError(
            "Target/reference cropped image sizes differ. For this gate, use "
            "same-size pathology patches so FLUX packed tokens align exactly. "
            f"target={target_original.size}, reference={reference_original.size}"
        )
    width, height = target_original.size

    target_original_path = output_dir / "target_original_cropped.png"
    reference_original_path = output_dir / "reference_original_cropped.png"
    target_original.save(target_original_path)
    reference_original.save(reference_original_path)
    # Backward-compatible alias for earlier STEP 1 artifact readers.
    target_original.save(output_dir / "original_cropped.png")

    print(f"Loading RF-Solver FLUX components for {args.name} on {torch_device}...")
    print(f"Local FLUX weight paths: {json.dumps(local_weight_paths, sort_keys=True)}")
    ae = rf["load_ae"](
        args.name,
        device="cpu" if args.offload else torch_device,
    )

    if args.offload:
        ae.encoder.to(torch_device)
    target_latents = encode_image(target_original, torch_device, ae)
    reference_latents = encode_image(reference_original, torch_device, ae)
    if args.offload:
        ae.cpu()
        maybe_empty_cuda_cache(torch_device)

    max_text_length = 256 if args.name == "flux-schnell" else 512
    text_encoder_paths = resolve_text_encoder_paths(args)
    if validate_text_encoder_paths(text_encoder_paths):
        t5, clip = load_local_text_encoders(
            device=torch_device,
            max_length=max_text_length,
            paths=text_encoder_paths,
        )
    elif args.allow_text_encoder_download:
        print(
            "Local FLUX text encoder paths were not found; falling back to "
            "RF-Solver-Edit's downloader because --allow-text-encoder-download is set."
        )
        t5 = rf["load_t5"](torch_device, max_length=max_text_length)
        clip = rf["load_clip"](torch_device)
    else:
        raise FileNotFoundError(
            "Local text encoder paths were not found, and downloads are disabled. "
            "Set FLUX_DIFFUSERS_ROOT=/data/huggingface/FLUX.1-dev if that directory "
            "contains text_encoder_2/tokenizer_2/text_encoder/tokenizer, or pass "
            "--t5-model-path --t5-tokenizer-path --clip-model-path --clip-tokenizer-path. "
            f"Resolved paths were: {text_encoder_paths}"
        )

    model = rf["load_flow_model"](
        args.name,
        device="cpu" if args.offload else torch_device,
    )

    print("Preparing target/reference conditioning...")
    prepared_target_source = rf["prepare"](t5, clip, target_latents, prompt=source_prompt)
    prepared_target_denoise = (
        dict(prepared_target_source)
        if target_prompt == source_prompt
        else rf["prepare"](t5, clip, target_latents, prompt=target_prompt)
    )
    prepared_reference = rf["prepare"](t5, clip, reference_latents, prompt=reference_prompt)
    image_token_count = ensure_same_packed_token_count(
        prepared_target_source,
        prepared_reference,
    )

    timesteps = rf["get_schedule"](
        args.num_inference_steps,
        image_token_count,
        shift=(args.name != "flux-schnell"),
    )
    inject_steps = min(
        args.inject_steps if args.inject_steps > 0 else DEFAULT_CROSS_INJECT_STEPS,
        args.num_inference_steps,
    )
    denoise_fn = rf["denoise"] if args.with_second_order else denoise_first_order

    regional_labels = build_regional_token_labels(
        mode=args.regional_mode,
        target_tissue_mask=resolved["target_tissue_mask"],
        reference_tissue_mask=resolved["reference_tissue_mask"],
        target_nuclei_mask=resolved["target_nuclei_mask"],
        reference_nuclei_mask=resolved["reference_nuclei_mask"],
        num_image_tokens=image_token_count,
        target_size=target_original.size,
        reference_size=reference_original.size,
    )
    token_mask_artifacts = save_regional_token_overlays(regional_labels, output_dir)

    bank = CrossImageFeatureBank(
        after_layer=args.cross_after_layer,
        image_token_count=image_token_count,
        strength=args.cross_image_strength,
        regional_labels=regional_labels,
        inject_after_t=args.inject_after_t,
    )

    if args.offload:
        t5.cpu()
        clip.cpu()
        maybe_empty_cuda_cache(torch_device)
        model.to(torch_device)

    def _base_info(feature_name: str) -> dict[str, Any]:
        return {
            "feature_path": str(output_dir / feature_name),
            "feature": {},
            "inject_step": inject_steps,
            "_cross_bank": bank,
        }

    print(
        "Running target inversion "
        f"({args.num_inference_steps} steps, inject_steps={inject_steps}, "
        f"second_order={args.with_second_order})..."
    )
    bank.phase = "target_inversion"
    target_noise, _ = denoise_fn(
        model,
        **prepared_target_source,
        timesteps=timesteps,
        guidance=1.0,
        inverse=True,
        info=_base_info("features_target"),
    )

    print("Running reference inversion and saving K/V features...")
    bank.phase = "reference_inversion"
    _reference_noise, _ = denoise_fn(
        model,
        **prepared_reference,
        timesteps=timesteps,
        guidance=1.0,
        inverse=True,
        info=_base_info("features_reference"),
    )

    print("Running target baseline reconstruction without cross-image injection...")
    bank.phase = "baseline_denoise"
    baseline_inputs = dict(prepared_target_denoise)
    baseline_inputs["img"] = target_noise.clone()
    baseline, _ = denoise_fn(
        model,
        **baseline_inputs,
        timesteps=timesteps,
        guidance=args.guidance,
        inverse=False,
        info={
            "feature_path": str(output_dir / "features_baseline"),
            "feature": {},
            "inject_step": 0,
        },
    )

    if args.offload:
        model.cpu()
        maybe_empty_cuda_cache(torch_device)
        ae.decoder.to(torch_device)

    print("Decoding baseline reconstruction...")
    baseline_image = decode_image(
        baseline,
        height=height,
        width=width,
        ae=ae,
        unpack=rf["unpack"],
        device=torch_device,
    )
    baseline_path = output_dir / "baseline_reconstruction.png"
    baseline_image.save(baseline_path)
    baseline_metrics = image_metrics(target_original, baseline_image)

    if args.offload:
        ae.cpu()
        maybe_empty_cuda_cache(torch_device)
        model.to(torch_device)
        maybe_empty_cuda_cache(torch_device)

    modes = ["v-only", "kv"] if args.cross_image_mode == "both" else [args.cross_image_mode]
    cross_artifacts: dict[str, dict[str, Any]] = {}
    cross_panels: list[tuple[str, Image.Image]] = [
        ("target_i0", target_original),
        ("reference", reference_original),
        ("baseline", baseline_image),
    ]

    for mode in modes:
        print(
            "Running cross-image denoise "
            f"(mode={mode}, regional={args.regional_mode}, strength={args.cross_image_strength})..."
        )
        if args.offload:
            model.to(torch_device)
        bank.phase = "cross_denoise"
        bank.denoise_mode = mode
        cross_inputs = dict(prepared_target_denoise)
        cross_inputs["img"] = target_noise.clone()
        cross, _ = denoise_fn(
            model,
            **cross_inputs,
            timesteps=timesteps,
            guidance=args.guidance,
            inverse=False,
            info=_base_info(f"features_cross_{mode.replace('-', '_')}"),
        )
        if args.offload:
            model.cpu()
            maybe_empty_cuda_cache(torch_device)
            ae.decoder.to(torch_device)
        cross_image = decode_image(
            cross,
            height=height,
            width=width,
            ae=ae,
            unpack=rf["unpack"],
            device=torch_device,
        )
        stem = mode_file_stem(mode, args.regional_mode)
        cross_path = output_dir / f"{stem}.png"
        cross_image.save(cross_path)
        aliases: list[str] = []
        if args.regional_mode == "none" and mode == "kv":
            alias = output_dir / "cross_kv_global.png"
            cross_image.save(alias)
            aliases.append(str(alias))
        if args.regional_mode == "none" and mode == "v-only":
            alias = output_dir / "cross_v_only.png"
            if alias != cross_path:
                cross_image.save(alias)
                aliases.append(str(alias))
        if args.regional_mode == "none" and mode == "kv":
            alias = output_dir / "cross_kv.png"
            if alias != cross_path:
                cross_image.save(alias)
                aliases.append(str(alias))
        if args.regional_mode != "none" and mode == "kv":
            alias = output_dir / f"cross_kv_regional_{args.regional_mode}.png"
            if alias != cross_path:
                cross_image.save(alias)
                aliases.append(str(alias))

        diff_vs_target = make_diff_image(target_original, cross_image)
        diff_path = output_dir / f"{stem}_vs_target_diff.png"
        diff_vs_target.save(diff_path)
        cross_metrics_target = image_metrics(target_original, cross_image)
        cross_metrics_reference = image_metrics(reference_original, cross_image)
        cross_artifacts[mode] = {
            "image": str(cross_path),
            "aliases": aliases,
            "diff_vs_target": str(diff_path),
            "metrics_vs_target": cross_metrics_target,
            "metrics_vs_reference": cross_metrics_reference,
        }
        cross_panels.append((stem, cross_image))
        if args.offload:
            ae.cpu()
            maybe_empty_cuda_cache(torch_device)

    if args.offload:
        model.cpu()
        maybe_empty_cuda_cache(torch_device)

    comparison = make_labeled_grid(cross_panels, columns=min(3, len(cross_panels)))
    comparison_path = output_dir / "comparison_cross_modes.png"
    comparison.save(comparison_path)

    feature_summary = bank.summary()
    summary = {
        "mode": "cross_image",
        "cross_image_mode": args.cross_image_mode,
        "regional_mode": args.regional_mode,
        "num_inference_steps": args.num_inference_steps,
        "with_second_order": args.with_second_order,
        "guidance": args.guidance,
        "inject_steps": inject_steps,
        "cross_after_layer": args.cross_after_layer,
        "cross_image_strength": args.cross_image_strength,
        "inject_after_t": args.inject_after_t,
        "source_prompt": source_prompt,
        "prompt_source": resolved["prompt_source"],
        "target_prompt": target_prompt,
        "reference_prompt": reference_prompt,
        "reference_prompt_source": resolved["reference_prompt_source"],
        "name": args.name,
        "device": args.device,
        "offload": args.offload,
        "target_image": str(image_path),
        "reference_image": str(reference_image_path),
        "metadata_path": str(args.metadata) if args.metadata is not None else None,
        "metadata_index": args.metadata_index if args.metadata is not None else None,
        "metadata_sample_id": args.sample_id,
        "metadata_record": metadata_summary(record),
        "reference_metadata_record": metadata_summary(reference_record),
        "reference_selection": resolved["reference_selection"],
        "mask_paths": {
            "target_tissue_mask": str(resolved["target_tissue_mask"]) if resolved["target_tissue_mask"] is not None else None,
            "reference_tissue_mask": str(resolved["reference_tissue_mask"]) if resolved["reference_tissue_mask"] is not None else None,
            "target_nuclei_mask": str(resolved["target_nuclei_mask"]) if resolved["target_nuclei_mask"] is not None else None,
            "reference_nuclei_mask": str(resolved["reference_nuclei_mask"]) if resolved["reference_nuclei_mask"] is not None else None,
        },
        "text_encoder_paths": {
            key: str(value) if value is not None else None
            for key, value in text_encoder_paths.items()
        },
        "crop": {
            "target": target_crop_info,
            "reference": reference_crop_info,
        },
        "token_counts": {
            "image_tokens": image_token_count,
            "expected_single_total_for_flux_dev": image_token_count + max_text_length,
        },
        "baseline_metrics_vs_target": baseline_metrics,
        "feature_summary": feature_summary,
        "artifacts": {
            "target_original_cropped": str(target_original_path),
            "reference_original_cropped": str(reference_original_path),
            "baseline_reconstruction": str(baseline_path),
            "comparison_cross_modes": str(comparison_path),
            "token_masks": token_mask_artifacts,
            "cross_outputs": cross_artifacts,
        },
        "runtime_seconds": round(time.perf_counter() - started_at, 3),
        "notes": (
            "STEP 3 acceptance is visual: target structure should stay aligned "
            "while texture/stain moves toward the reference. STEP 4 regional mode "
            "uses token-level masks over image tokens only; high-frequency "
            "protection is intentionally not implemented in this first pass."
        ),
    }
    summary_path = output_dir / "cross_feature_summary.json"
    write_metrics(summary_path, summary)
    if args.save_feature_debug or args.debug_features:
        (output_dir / "cross_feature_events.json").write_text(
            json.dumps(bank.events, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"Saved cross-image artifacts to {output_dir}")
    print(
        "Cross summary | baseline PSNR={psnr} | baseline SSIM={ssim:.6f} | "
        "ref feature count={count} | missing keys={missing}".format(
            psnr=baseline_metrics["psnr"],
            ssim=baseline_metrics["ssim"],
            count=feature_summary["reference"]["feature_count"],
            missing=feature_summary["missing_key_count"],
        )
    )
    return 0


def run_reconstruction(args: argparse.Namespace) -> int:
    if args.cross_image:
        return run_cross_image_reconstruction(args)

    resolved = resolve_run_inputs(args)
    image_path: Path = resolved["image_path"]
    output_dir: Path = resolved["output_dir"]
    source_prompt: str = resolved["source_prompt"]
    record: dict[str, Any] | None = resolved["record"]

    prepend_rf_solver_src(args.rf_solver_root)
    rf = import_rf_solver_modules()
    if args.debug_features:
        install_feature_debug_hook(rf["SingleStreamBlock"])
    if args.name not in rf["configs"]:
        available = ", ".join(sorted(rf["configs"].keys()))
        raise ValueError(f"Unknown --name {args.name!r}; available names: {available}")
    local_weight_paths = configure_local_flux_weight_paths(rf, args)

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch_device = torch.device(args.device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)
    started_at = time.perf_counter()

    original, crop_info = crop_to_multiple(load_rgb_image(image_path))
    width, height = original.size
    original_path = output_dir / "original_cropped.png"
    original.save(original_path)

    print(f"Loading RF-Solver FLUX components for {args.name} on {torch_device}...")
    print(f"Local FLUX weight paths: {json.dumps(local_weight_paths, sort_keys=True)}")
    ae = rf["load_ae"](
        args.name,
        device="cpu" if args.offload else torch_device,
    )

    if args.offload:
        ae.encoder.to(torch_device)
    latents = encode_image(original, torch_device, ae)
    if args.offload:
        ae.cpu()
        maybe_empty_cuda_cache(torch_device)

    if args.vae_roundtrip_only:
        if args.offload:
            ae.decoder.to(torch_device)
        reconstruction_image = decode_latents(latents, ae=ae, device=torch_device)
        reconstruction_path = output_dir / "reconstruction.png"
        reconstruction_image.save(reconstruction_path)
        diff_image = make_diff_image(original, reconstruction_image)
        diff_path = output_dir / "diff.png"
        diff_image.save(diff_path)
        comparison_image = make_comparison_image(original, reconstruction_image, diff_image)
        comparison_path = output_dir / "comparison.png"
        comparison_image.save(comparison_path)
        metrics = image_metrics(original, reconstruction_image)
        metrics.update(
            {
                "gate": "diagnostic_vae_roundtrip_only",
                "mode": "vae_roundtrip_only",
                "input_image": str(image_path),
                "metadata_path": str(args.metadata) if args.metadata is not None else None,
                "metadata_index": args.metadata_index if args.metadata is not None else None,
                "metadata_sample_id": args.sample_id,
                "metadata_image_field": args.image_field if args.metadata is not None else None,
                "metadata_record": metadata_summary(record),
                "crop": crop_info,
                "artifacts": {
                    "original_cropped": str(original_path),
                    "reconstruction": str(reconstruction_path),
                    "diff": str(diff_path),
                    "comparison": str(comparison_path),
                },
                "runtime_seconds": round(time.perf_counter() - started_at, 3),
                "notes": (
                    "This diagnostic isolates FLUX VAE encode/decode artifacts. "
                    "If the same bottom stripe appears here, it is not caused by "
                    "RF-Solver denoising."
                ),
            }
        )
        metrics_path = output_dir / "metrics.json"
        write_metrics(metrics_path, metrics)
        print(f"Saved VAE roundtrip artifacts to {output_dir}")
        print(
            "VAE roundtrip | PSNR={psnr} | SSIM={ssim:.6f}".format(
                psnr=metrics["psnr"],
                ssim=metrics["ssim"],
            )
        )
        return 0

    max_text_length = 256 if args.name == "flux-schnell" else 512
    text_encoder_paths = resolve_text_encoder_paths(args)
    if validate_text_encoder_paths(text_encoder_paths):
        t5, clip = load_local_text_encoders(
            device=torch_device,
            max_length=max_text_length,
            paths=text_encoder_paths,
        )
    elif args.allow_text_encoder_download:
        print(
            "Local FLUX text encoder paths were not found; falling back to "
            "RF-Solver-Edit's downloader because --allow-text-encoder-download is set."
        )
        t5 = rf["load_t5"](torch_device, max_length=max_text_length)
        clip = rf["load_clip"](torch_device)
    else:
        raise FileNotFoundError(
            "Local text encoder paths were not found, and downloads are disabled. "
            "Set FLUX_DIFFUSERS_ROOT=/data/huggingface/FLUX.1-dev if that directory "
            "contains text_encoder_2/tokenizer_2/text_encoder/tokenizer, or pass "
            "--t5-model-path --t5-tokenizer-path --clip-model-path --clip-tokenizer-path. "
            f"Resolved paths were: {text_encoder_paths}"
        )
    model = rf["load_flow_model"](
        args.name,
        device="cpu" if args.offload else torch_device,
    )

    print("Preparing text/image conditioning...")
    target_prompt = source_prompt if args.target_prompt is None else args.target_prompt
    is_reconstruction_mode = target_prompt == source_prompt
    prepared = rf["prepare"](t5, clip, latents, prompt=source_prompt)
    prepared_target = (
        dict(prepared)
        if is_reconstruction_mode
        else rf["prepare"](t5, clip, latents, prompt=target_prompt)
    )
    timesteps = rf["get_schedule"](
        args.num_inference_steps,
        prepared["img"].shape[1],
        shift=(args.name != "flux-schnell"),
    )
    inject_steps = min(args.inject_steps, args.num_inference_steps)
    info = {
        "feature_path": str(output_dir / "features"),
        "feature": {},
        "inject_step": inject_steps,
    }
    denoise_fn = rf["denoise"] if args.with_second_order else denoise_first_order

    if args.offload:
        t5.cpu()
        clip.cpu()
        maybe_empty_cuda_cache(torch_device)
        model.to(torch_device)

    print(
        "Running inversion "
        f"({args.num_inference_steps} steps, second_order={args.with_second_order})..."
    )
    noise, info = denoise_fn(
        model,
        **prepared,
        timesteps=timesteps,
        guidance=1.0,
        inverse=True,
        info=info,
    )
    debug_summaries: dict[str, Any] = {}
    if args.debug_features:
        debug_summaries["inversion_save"] = print_debug_summary(
            info.get("_debug_events", []),
            "inversion_save",
            args.debug_max_events,
        )

    print("Running reconstruction denoise...")
    reconstruction_inputs = dict(prepared_target)
    reconstruction_inputs["img"] = noise
    reconstruction, _ = denoise_fn(
        model,
        **reconstruction_inputs,
        timesteps=timesteps,
        guidance=args.guidance,
        inverse=False,
        info=info,
    )
    if args.debug_features:
        debug_summaries["denoise_load"] = print_debug_summary(
            info.get("_debug_events", []),
            "denoise_load",
            args.debug_max_events,
        )
        (output_dir / "feature_debug_events.json").write_text(
            json.dumps(info.get("_debug_events", []), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    if args.offload:
        model.cpu()
        maybe_empty_cuda_cache(torch_device)
        ae.decoder.to(torch_device)

    print("Decoding reconstruction without watermark or NSFW filtering...")
    reconstruction_image = decode_image(
        reconstruction,
        height=height,
        width=width,
        ae=ae,
        unpack=rf["unpack"],
        device=torch_device,
    )
    reconstruction_path = output_dir / "reconstruction.png"
    reconstruction_image.save(reconstruction_path)

    diff_image = make_diff_image(original, reconstruction_image)
    diff_path = output_dir / "diff.png"
    diff_image.save(diff_path)

    comparison_image = make_comparison_image(original, reconstruction_image, diff_image)
    comparison_path = output_dir / "comparison.png"
    comparison_image.save(comparison_path)

    metrics = image_metrics(original, reconstruction_image)
    psnr_value = metric_value_for_threshold(metrics["psnr"])
    ssim_value = float(metrics["ssim"])
    gate_pass = psnr_value > args.psnr_threshold or ssim_value > args.ssim_threshold
    gate = "green_light" if gate_pass else "red_light"
    if not is_reconstruction_mode:
        gate = "not_applicable_edit_debug"
    metrics.update(
        {
            "gate": gate,
            "mode": "reconstruction" if is_reconstruction_mode else "edit_debug",
            "psnr_threshold": args.psnr_threshold,
            "ssim_threshold": args.ssim_threshold,
            "num_inference_steps": args.num_inference_steps,
            "with_second_order": args.with_second_order,
            "guidance": args.guidance,
            "source_prompt": source_prompt,
            "prompt_source": resolved["prompt_source"],
            "target_prompt": target_prompt,
            "name": args.name,
            "device": args.device,
            "offload": args.offload,
            "inject_steps": inject_steps,
            "input_image": str(image_path),
            "metadata_path": str(args.metadata) if args.metadata is not None else None,
            "metadata_index": args.metadata_index if args.metadata is not None else None,
            "metadata_sample_id": args.sample_id,
            "metadata_image_field": args.image_field if args.metadata is not None else None,
            "metadata_record": metadata_summary(record),
            "text_encoder_paths": {
                key: str(value) if value is not None else None
                for key, value in text_encoder_paths.items()
            },
            "crop": crop_info,
            "artifacts": {
                "original_cropped": str(original_path),
                "reconstruction": str(reconstruction_path),
                "diff": str(diff_path),
                "comparison": str(comparison_path),
            },
            "runtime_seconds": round(time.perf_counter() - started_at, 3),
            "feature_debug": debug_summaries,
            "notes": (
                "If this gate is red or the visual comparison is blurry/lost "
                "texture, stop before STEP 2 and discuss RF-Inversion, "
                "OT-Inversion, or abandoning this path."
            ),
        }
    )
    metrics_path = output_dir / "metrics.json"
    write_metrics(metrics_path, metrics)

    print(f"Saved reconstruction artifacts to {output_dir}")
    print(
        "Gate: {gate} | PSNR={psnr} | SSIM={ssim:.6f}".format(
            gate=metrics["gate"],
            psnr=metrics["psnr"],
            ssim=metrics["ssim"],
        )
    )
    if is_reconstruction_mode and not gate_pass and args.fail_on_threshold:
        return 2
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.metric_self_test:
        return run_metric_self_test()
    if args.metadata_self_test:
        return run_metadata_self_test()
    if args.cross_self_test:
        return run_cross_self_test()
    return run_reconstruction(args)


if __name__ == "__main__":
    raise SystemExit(main())

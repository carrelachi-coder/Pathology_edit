#!/usr/bin/env python3
"""Replay frozen Cross Stage1 images with lower same-WSI identity gain."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.pix2pix_transfer.inference import (  # noqa: E402
    load_pix2pix_postprocessor,
    run_pix2pix_postprocess,
)
from phase3_mask_edit.audit import (  # noqa: E402
    dataset_native_metric_class_ids,
    source_relative_tissue_metrics,
    to_coarse_mask,
)
from scripts.replay_product_quality_evaluator import (  # noqa: E402
    _find_generation_inputs,
    _load_mask,
    _load_probabilities,
    _optional_array,
    _read_json,
)


PALETTE = np.asarray(
    [
        (245, 245, 245),
        (196, 52, 52),
        (56, 158, 70),
        (232, 195, 36),
        (55, 105, 210),
        (224, 107, 176),
        (44, 166, 174),
        (135, 135, 135),
    ],
    dtype=np.uint8,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pix2pix-checkpoint", type=Path, required=True)
    parser.add_argument("--segmentator-release", type=Path, required=True)
    parser.add_argument("--stage1-segmentator-root", type=Path, required=True)
    parser.add_argument("--gains", type=float, nargs="+", default=(1.15, 1.0))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    candidates = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise ValueError("candidate manifest is empty")
    if any(gain < 1.0 or gain > 1.5 for gain in args.gains):
        raise ValueError("diagnostic gains must be in [1.0, 1.5]")

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    bundle = load_pix2pix_postprocessor(
        args.pix2pix_checkpoint,
        device=device,
        torch_dtype=dtype,
    )
    generated: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        case_id = str(candidate["case_id"])
        case_dir = args.output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        stage1_path = Path(str(candidate["stage1"]))
        agentic_dir = Path(str(candidate["agentic_dir"]))
        workflow = _read_json(agentic_dir / "pipeline_summary.json")
        inputs = _find_generation_inputs(agentic_dir, workflow)
        stage1 = Image.open(stage1_path).convert("RGB")
        for gain in args.gains:
            gain_id = _gain_id(gain)
            output_dir = case_dir / gain_id
            output_dir.mkdir(parents=True, exist_ok=True)
            adjusted = replace(
                bundle,
                config=replace(
                    bundle.config,
                    same_wsi_identity_low_support_max_gain=float(gain),
                ),
            )
            output_image, metadata = run_pix2pix_postprocess(
                bundle=adjusted,
                i0_image=stage1,
                reference_image_path=inputs["reference_image"],
                target_tissue_mask_path=inputs["target_tissue_mask"],
                target_nuclei_mask_path=inputs["target_nuclei_mask"],
                reference_tissue_mask_path=inputs["reference_tissue_mask"],
                reference_nuclei_mask_path=inputs["reference_nuclei_mask"],
                image_size=stage1.width,
                device=device,
                torch_dtype=dtype,
                low_stain_protection_region_path=(
                    agentic_dir / "generation_change_region.png"
                ),
                low_stain_protection_mask_output_path=(
                    output_dir / "low_stain_protection_mask.png"
                ),
                unprotected_output_path=(
                    output_dir / "generated_image_unprotected.png"
                ),
            )
            image_path = output_dir / "generated_image.png"
            output_image.save(image_path)
            _write_json(
                output_dir / "pix2pix_metadata.json",
                {
                    "case_id": case_id,
                    "maximum_identity_gain": float(gain),
                    "stage1": str(stage1_path),
                    "metadata": metadata,
                },
            )
            generated.append(
                {
                    "case_id": case_id,
                    "profile": str(candidate["profile"]),
                    "gain": float(gain),
                    "image_path": image_path,
                    "output_dir": output_dir,
                    "agentic_dir": agentic_dir,
                    "stage1": stage1_path,
                }
            )
        print(f"[{index}/{len(candidates)}] generated {case_id}", flush=True)

    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _run_segmentator(generated, release_path=args.segmentator_release, device=device)

    results = []
    by_case = {str(item["case_id"]): item for item in candidates}
    for case_id, candidate in by_case.items():
        case_generated = [item for item in generated if item["case_id"] == case_id]
        result = _evaluate_case(
            candidate=candidate,
            generated=case_generated,
            output_dir=args.output / case_id,
            stage1_segmentator_root=args.stage1_segmentator_root,
        )
        results.append(result)
    summary = _summarize(results, gains=args.gains)
    _write_json(args.output / "low_gain_retry_summary.json", summary)
    _build_overview(results, args.output / "low_gain_retry_overview.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _run_segmentator(
    generated: Sequence[Mapping[str, Any]],
    *,
    release_path: Path,
    device: Any,
) -> None:
    import torch
    import torchvision.transforms.functional as TF

    from segmentator.data import normalize_image_tensor
    from segmentator.inference import (
        load_checkpoint,
        normalized_entropy,
        save_prediction,
        save_probability_tensor,
    )
    from segmentator.release import load_segmentator_release, release_model_kwargs

    release = load_segmentator_release(release_path, verify_checkpoint=True)
    model = load_checkpoint(
        release["checkpoint"],
        **release_model_kwargs(release),
    ).to(device)
    for index, item in enumerate(generated, start=1):
        output_dir = Path(item["output_dir"]) / "segmentator"
        output_dir.mkdir(parents=True, exist_ok=True)
        image = normalize_image_tensor(
            TF.to_tensor(Image.open(item["image_path"]).convert("RGB"))
        ).to(device)
        with torch.inference_mode():
            outputs = model(image.unsqueeze(0))
        save_prediction(outputs["pred"][0], output_dir / "coarse_mask.png")
        save_probability_tensor(
            outputs["probs"],
            output_dir / "coarse_probabilities.npz",
            class_ids=tuple(range(outputs["probs"].shape[1])),
        )
        entropy = normalized_entropy(outputs["probs"])[0]
        np.save(
            output_dir / "entropy.npy",
            entropy.detach().cpu().numpy().astype(np.float16),
        )
        print(f"[{index}/{len(generated)}] segmented {item['case_id']}", flush=True)


def _evaluate_case(
    *,
    candidate: Mapping[str, Any],
    generated: Sequence[Mapping[str, Any]],
    output_dir: Path,
    stage1_segmentator_root: Path,
) -> dict[str, Any]:
    case_id = str(candidate["case_id"])
    profile = str(candidate["profile"])
    agentic_dir = Path(str(candidate["agentic_dir"]))
    workflow = _read_json(agentic_dir / "pipeline_summary.json")
    inputs = _find_generation_inputs(agentic_dir, workflow)
    source_mask = to_coarse_mask(_load_mask(inputs["reference_tissue_mask"]))
    target_mask = to_coarse_mask(_load_mask(inputs["target_tissue_mask"]))
    semantic_region = _load_mask(agentic_dir / "semantic_change_region.png") > 0
    generation_region = _load_mask(agentic_dir / "generation_change_region.png") > 0
    source_dir = agentic_dir / "source_verification"
    class_ids = dataset_native_metric_class_ids(profile, level="coarse")

    def metrics(prediction_dir: Path) -> dict[str, Any]:
        return source_relative_tissue_metrics(
            source_mask=source_mask,
            target_mask=target_mask,
            source_prediction=_load_mask(source_dir / "coarse_mask.png"),
            generated_prediction=_load_mask(prediction_dir / "coarse_mask.png"),
            source_probabilities=_load_probabilities(
                source_dir / "coarse_probabilities.npz"
            ),
            generated_probabilities=_load_probabilities(
                prediction_dir / "coarse_probabilities.npz"
            ),
            class_ids=class_ids,
            source_entropy=_optional_array(source_dir / "entropy.npy"),
            generated_entropy=_optional_array(prediction_dir / "entropy.npy"),
            semantic_change_region=semantic_region,
            preservation_exclusion_region=generation_region,
        )

    stage1_dir = stage1_segmentator_root / case_id
    attempt_dir = Path(str(candidate["stage1"])).parents[1]
    original_dir = attempt_dir / "verification"
    measurements = {
        "stage1": _compact_metrics(metrics(stage1_dir)),
        "gain_1_50": _compact_metrics(metrics(original_dir)),
    }
    image_paths = {
        "source": Path(inputs["reference_image"]),
        "target_mask": Path(inputs["target_tissue_mask"]),
        "stage1": Path(str(candidate["stage1"])),
        "gain_1_50": attempt_dir / "generated_image.png",
    }
    prediction_paths = {
        "source": source_dir / "coarse_mask.png",
        "stage1": stage1_dir / "coarse_mask.png",
        "gain_1_50": original_dir / "coarse_mask.png",
    }
    for item in generated:
        gain_id = _gain_id(float(item["gain"]))
        measurements[gain_id] = _compact_metrics(
            metrics(Path(item["output_dir"]) / "segmentator")
        )
        image_paths[gain_id] = Path(item["image_path"])
        prediction_paths[gain_id] = (
            Path(item["output_dir"]) / "segmentator" / "coarse_mask.png"
        )

    original_mean_gain = _original_mean_gain(workflow)
    result = {
        "case_id": case_id,
        "profile": profile,
        "primitive": str(candidate["primitive"]),
        "original_mean_tissue_gain": original_mean_gain,
        "measurements": measurements,
    }
    _write_json(output_dir / "comparison.json", result)
    panel = _build_case_panel(
        result=result,
        image_paths=image_paths,
        prediction_paths=prediction_paths,
        target_mask_path=Path(inputs["target_tissue_mask"]),
        semantic_region=semantic_region,
    )
    panel_path = output_dir / "comparison_panel.png"
    panel.save(panel_path)
    result["panel"] = str(panel_path)
    return result


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    changed = dict(metrics["changed_region"])
    preservation = dict(metrics["preservation"])
    values = [
        _first_number(
            changed,
            "appearance_calibrated_target_probability_gain",
            "target_probability_gain",
        ),
        _first_number(
            changed,
            "appearance_calibrated_source_probability_suppression",
            "source_probability_suppression",
        ),
        _first_number(
            changed,
            "appearance_calibrated_soft_margin_gain",
            "soft_margin_gain",
        ),
    ]
    active = [value for value in values if value is not None]
    direction_terms = []
    for weight, value in zip((0.4, 0.4, 0.2), values):
        if value is not None:
            direction_terms.append((weight, _direction_score(value)))
    return {
        "semantic_direction_score": sum(w * v for w, v in direction_terms)
        / sum(w for w, _ in direction_terms),
        "mean_signed_semantic_gain": float(sum(active) / len(active)),
        "target_probability_gain": float(values[0] or 0.0),
        "source_probability_suppression": float(values[1] or 0.0),
        "margin_gain": float(values[2] or 0.0),
        "preservation_drift": float(
            preservation.get("appearance_calibrated_prediction_drift_U_far")
            if preservation.get("appearance_calibration_applicable")
            else preservation.get("prediction_relative_drift_U_far", 1.0)
        ),
    }


def _summarize(
    results: Sequence[Mapping[str, Any]],
    *,
    gains: Sequence[float],
) -> dict[str, Any]:
    gain_ids = [_gain_id(gain) for gain in gains]
    fixed = {gain_id: 0 for gain_id in gain_ids}
    direction_improved = {gain_id: 0 for gain_id in gain_ids}
    drift_improved = {gain_id: 0 for gain_id in gain_ids}
    for result in results:
        measurements = result["measurements"]
        baseline = measurements["gain_1_50"]
        for gain_id in gain_ids:
            current = measurements[gain_id]
            fixed[gain_id] += current["semantic_direction_score"] >= 0.60
            direction_improved[gain_id] += (
                current["mean_signed_semantic_gain"]
                > baseline["mean_signed_semantic_gain"]
            )
            drift_improved[gain_id] += (
                current["preservation_drift"] < baseline["preservation_drift"]
            )
    return {
        "schema_version": 1,
        "experiment": "frozen_stage1_pix2pix_low_identity_gain",
        "case_count": len(results),
        "formal_product_gain_unchanged": 1.5,
        "diagnostic_gains": list(gains),
        "direction_threshold": 0.60,
        "direction_fixed_counts": fixed,
        "mean_direction_improved_counts": direction_improved,
        "preservation_drift_improved_counts": drift_improved,
        "results": list(results),
    }


def _build_case_panel(
    *,
    result: Mapping[str, Any],
    image_paths: Mapping[str, Path],
    prediction_paths: Mapping[str, Path],
    target_mask_path: Path,
    semantic_region: np.ndarray,
) -> Image.Image:
    order = ["source", "target_mask", "stage1", "gain_1_50"] + sorted(
        key for key in image_paths if key.startswith("gain_") and key != "gain_1_50"
    )
    tile = 512
    header = 68
    canvas = Image.new("RGB", (tile * len(order), header + tile * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    target_mask = to_coarse_mask(_load_mask(target_mask_path))
    for column, key in enumerate(order):
        x = column * tile
        if key == "target_mask":
            top = _mask_image(target_mask)
            bottom = top.copy()
            label = "Approved target mask"
        else:
            top = Image.open(image_paths[key]).convert("RGB").resize((tile, tile))
            top = _outline_region(top, semantic_region)
            prediction = to_coarse_mask(_load_mask(prediction_paths[key]))
            bottom = _overlay_mask(top, prediction)
            label = key.replace("_", " ")
        canvas.paste(top, (x, header))
        canvas.paste(bottom, (x, header + tile))
        measurement = result["measurements"].get(key)
        text = label
        if measurement:
            text += (
                f"\ndir={measurement['semantic_direction_score']:.2f} "
                f"mean={measurement['mean_signed_semantic_gain']:+.4f}"
                f"\ndrift={measurement['preservation_drift']:.3f}"
            )
        draw.multiline_text((x + 6, 5), text, fill="black", font=font, spacing=2)
    return canvas


def _build_overview(results: Sequence[Mapping[str, Any]], output: Path) -> None:
    thumbnails = []
    for result in results:
        panel = Image.open(result["panel"]).convert("RGB")
        panel.thumbnail((1600, 360), Image.Resampling.LANCZOS)
        thumbnails.append((str(result["case_id"]), panel.copy()))
    width = max(image.width for _, image in thumbnails)
    row_height = max(image.height for _, image in thumbnails) + 24
    canvas = Image.new("RGB", (width, row_height * len(thumbnails)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (case_id, image) in enumerate(thumbnails):
        y = index * row_height
        draw.text((4, y + 4), case_id, fill="black", font=font)
        canvas.paste(image, (0, y + 24))
    canvas.save(output)


def _mask_image(mask: np.ndarray) -> Image.Image:
    safe = np.where((mask >= 0) & (mask < len(PALETTE)), mask, 7)
    return Image.fromarray(PALETTE[safe.astype(np.int64)], mode="RGB")


def _overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    color = _mask_image(mask).resize(image.size)
    return Image.blend(image.convert("RGB"), color, 0.42)


def _outline_region(image: Image.Image, region: np.ndarray) -> Image.Image:
    region = np.asarray(region, dtype=bool)
    outline = binary_dilation(region, iterations=2) ^ binary_erosion(
        region,
        iterations=1,
        border_value=0,
    )
    array = np.asarray(image).copy()
    array[outline] = np.asarray((255, 220, 0), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _original_mean_gain(workflow: Mapping[str, Any]) -> float | None:
    for attempt in workflow.get("attempts") or ():
        artifact = dict(attempt.get("artifact") or {})
        metadata = dict(artifact.get("metadata") or {})
        cross = dict(metadata.get("cross_v1") or {})
        pix2pix = dict(cross.get("pix2pix_v2") or {})
        identity = dict(pix2pix.get("same_wsi_identity") or {})
        value = identity.get("mean_tissue_gain")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _gain_id(gain: float) -> str:
    return f"gain_{gain:.2f}".replace(".", "_")


def _direction_score(value: float, epsilon: float = 1e-4) -> float:
    if value > epsilon:
        return 1.0
    if value < -epsilon:
        return 0.0
    return 0.5


def _first_number(values: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Visual audit artifacts for online pathology-edit self-auditing."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_config.unified_labels import UNIFIED_COLOR_MAP

from .labels import to_coarse_mask
from .metrics import build_edit_regions


TILE_SIZE = 256
PANEL_COLUMNS = 4


def build_online_audit_deck(
    *,
    manifest: Iterable[dict[str, Any]],
    run_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in manifest:
        condition_dir = run_root / record["condition_id"]
        panel_path, metadata = _build_condition_panel(
            record=record,
            condition_dir=condition_dir,
            output_path=panel_dir / f"{record['condition_id']}.png",
        )
        rows.append(
            {
                **metadata,
                "panel": str(panel_path),
                "panel_relative": str(panel_path.relative_to(output_dir)),
            }
        )
    contact_path = _build_contact_sheet(
        rows, output_dir / "online_agent_canary_contact_sheet.png"
    )
    html_path = _write_html(rows, output_dir / "index.html")
    comparison = _p1_comparison(rows)
    comparison_path = output_dir / "p1_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "condition_count": len(rows),
        "all_conditions_visualized": len(rows) == 18,
        "contact_sheet": str(contact_path),
        "html": str(html_path),
        "p1_comparison": str(comparison_path),
        "p1_recommendation": comparison["recommendation"],
    }
    (output_dir / "audit_deck_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _build_condition_panel(
    *,
    record: dict[str, Any],
    condition_dir: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    source_rgb = _rgb(record["reference_image_path"])
    source_fine = _mask(record["reference_tissue_mask_path"])
    target_fine = _mask(record["target_tissue_mask_path"])
    source_coarse = to_coarse_mask(source_fine)
    target_coarse = to_coarse_mask(target_fine)
    semantic = _binary(record["semantic_change_region_path"])
    generation = _binary(record["generation_change_region_path"])
    workflow = _json_or_empty(condition_dir / "agentic_workflow.json")
    summary = _json_or_empty(condition_dir / "pipeline_summary.json")
    selected = workflow.get("selected_attempt") or {}
    selected_artifact = selected.get("artifact") or {}
    selected_image_path = selected_artifact.get("image_path")
    generated_rgb = (
        _rgb(selected_image_path)
        if selected_image_path and Path(selected_image_path).is_file()
        else _placeholder("no selected image")
    )
    verification_dir = (
        Path(selected_image_path).parent / "verification"
        if selected_image_path
        else condition_dir / "missing_verification"
    )
    source_verification = condition_dir / "source_verification"
    source_prediction = _mask_or_none(source_verification / "coarse_mask.png")
    raw_prediction = _mask_or_none(verification_dir / "coarse_mask_raw.png")
    p1_prediction = _mask_or_none(verification_dir / "coarse_mask_p1.png")
    if raw_prediction is None:
        raw_prediction = _mask_or_none(verification_dir / "coarse_mask.png")
    if p1_prediction is None:
        p1_prediction = raw_prediction
    p1_changed = (
        raw_prediction != p1_prediction
        if raw_prediction is not None and p1_prediction is not None
        else np.zeros(source_coarse.shape, dtype=bool)
    )
    source_probabilities = _probabilities_or_none(
        source_verification / "coarse_probabilities.npz"
    )
    generated_probabilities = _probabilities_or_none(
        verification_dir / "coarse_probabilities.npz"
    )
    generated_entropy = _array_or_none(verification_dir / "entropy.npy")
    fine_prediction = _mask_or_none(verification_dir / "fine_mask.png")
    fine_entropy = _array_or_none(verification_dir / "fine_entropy.npy")
    predicted_nuclei = _mask_or_none(
        verification_dir / "predicted_nuclei_mask.png"
    )
    target_nuclei = _mask_or_none(record["target_nuclei_mask_path"])
    audit = _json_or_empty(verification_dir / "online_semantic_audit.json")
    regions = build_edit_regions(
        source_coarse,
        target_coarse,
        semantic_change_region=semantic,
    )
    drift = (
        raw_prediction != source_prediction
        if raw_prediction is not None and source_prediction is not None
        else np.zeros(source_coarse.shape, dtype=bool)
    )
    mismatch = (
        raw_prediction != target_coarse
        if raw_prediction is not None
        else np.zeros(source_coarse.shape, dtype=bool)
    )
    attempts = workflow.get("attempts") or []
    attempt_images = []
    for attempt in attempts[:2]:
        artifact = attempt.get("artifact") or {}
        path = artifact.get("image_path")
        attempt_images.append(
            _rgb(path) if path and Path(path).is_file() else _placeholder("missing")
        )
    while len(attempt_images) < 2:
        attempt_images.append(_placeholder("not run"))

    tiles = [
        ("Source RGB", source_rgb),
        ("Selected generated RGB", generated_rgb),
        ("Attempt 1", attempt_images[0]),
        ("Attempt 2", attempt_images[1]),
        ("Source fine condition", _colorize(source_fine)),
        ("Target fine condition", _colorize(target_fine)),
        ("Semantic region R", _overlay(source_rgb, semantic, (240, 40, 40))),
        ("Generation region G", _overlay(source_rgb, generation, (40, 160, 240))),
        ("Source Segmentator", _colorize_or_placeholder(source_prediction)),
        ("Generated raw", _colorize_or_placeholder(raw_prediction)),
        ("Generated P1 shadow", _colorize_or_placeholder(p1_prediction)),
        ("P1 changed pixels", _overlay(generated_rgb, p1_changed, (255, 220, 0))),
        (
            "Source top1 confidence",
            _heatmap_or_placeholder(_max_probability(source_probabilities)),
        ),
        (
            "Generated top1 confidence",
            _heatmap_or_placeholder(_max_probability(generated_probabilities)),
        ),
        ("Generated coarse entropy", _heatmap_or_placeholder(generated_entropy)),
        ("Generated fine entropy", _heatmap_or_placeholder(fine_entropy)),
        ("Generated fine prediction", _colorize_or_placeholder(fine_prediction)),
        ("Boundary B", _overlay(source_rgb, regions["B"], (255, 150, 0))),
        ("Unchanged far U_far", _overlay(source_rgb, regions["U_far"], (40, 190, 80))),
        ("Prediction drift", _overlay(generated_rgb, drift, (255, 0, 220))),
        ("Target mismatch", _overlay(generated_rgb, mismatch, (255, 50, 50))),
        ("Target nuclei", _nuclei_or_placeholder(target_nuclei)),
        ("Predicted CellViT", _nuclei_or_placeholder(predicted_nuclei)),
        (
            "Semantic != generation",
            _overlay(source_rgb, generation ^ semantic, (0, 240, 240)),
        ),
    ]
    header = (
        f"{record['condition_id']} | {record['organ']} / {record['profile']} | "
        f"{record['route_stratum']} | {record['canary_scenario']} | "
        f"status={summary.get('status', workflow.get('status', 'missing'))}"
    )
    panel = _grid(tiles, header=header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    raw_metrics = audit.get("raw_metrics") or {}
    p1_metrics = audit.get("p1_metrics") or {}
    raw_changed = raw_metrics.get("changed_region") or {}
    p1_changed_metrics = p1_metrics.get("changed_region") or {}
    raw_preservation = raw_metrics.get("preservation") or {}
    p1_preservation = p1_metrics.get("preservation") or {}
    p1_meta = audit.get("p1") or {}
    return output_path, {
        "condition_id": record["condition_id"],
        "sample_id": record["sample_id"],
        "organ": record["organ"],
        "profile": record["profile"],
        "route_stratum": record["route_stratum"],
        "scenario": record["canary_scenario"],
        "status": summary.get("status", workflow.get("status", "missing")),
        "attempt_count": len(attempts),
        "selected_mode": selected.get("requested_mode"),
        "raw_changed_accuracy": raw_changed.get("accuracy"),
        "p1_changed_accuracy": p1_changed_metrics.get("accuracy"),
        "raw_changed_macro_miou": raw_changed.get("macro_miou"),
        "p1_changed_macro_miou": p1_changed_metrics.get("macro_miou"),
        "raw_drift_u_far": raw_preservation.get(
            "prediction_relative_drift_U_far"
        ),
        "p1_drift_u_far": p1_preservation.get(
            "prediction_relative_drift_U_far"
        ),
        "p1_changed_pixels": p1_meta.get("changed_pixels", 0),
        "p1_operation_count": p1_meta.get("operation_count", 0),
        "semantic_generation_regions_equal": record[
            "semantic_generation_regions_equal"
        ],
        "fine_metrics": audit.get("fine_metrics"),
    }


def _p1_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = []
    for row in rows:
        deltas.append(
            {
                "condition_id": row["condition_id"],
                "changed_accuracy_delta": _delta(
                    row.get("p1_changed_accuracy"),
                    row.get("raw_changed_accuracy"),
                ),
                "changed_macro_miou_delta": _delta(
                    row.get("p1_changed_macro_miou"),
                    row.get("raw_changed_macro_miou"),
                ),
                "drift_u_far_delta": _delta(
                    row.get("p1_drift_u_far"),
                    row.get("raw_drift_u_far"),
                ),
                "changed_pixels": row.get("p1_changed_pixels", 0),
                "operation_count": row.get("p1_operation_count", 0),
            }
        )
    active = [row for row in deltas if int(row["changed_pixels"] or 0) > 0]
    harmful = [
        row
        for row in active
        if (row["changed_macro_miou_delta"] or 0.0) < -0.002
        or (row["drift_u_far_delta"] or 0.0) > 0.002
    ]
    beneficial = [
        row
        for row in active
        if (row["changed_macro_miou_delta"] or 0.0) > 0.002
        or (row["drift_u_far_delta"] or 0.0) < -0.002
    ]
    if not active:
        decision = "keep_shadow_insufficient_trigger_coverage"
    elif harmful:
        decision = "revise_policy_keep_shadow"
    elif beneficial:
        decision = "retain_candidate_and_calibrate_on_60_keep_shadow"
    else:
        decision = "retain_candidate_no_material_canary_effect_keep_shadow"
    return {
        "schema_version": 1,
        "policy_candidate": "conservative-island-p1-v1",
        "canary_role": "interface_and_failure_mode_validation_only",
        "formal_threshold_selection_allowed": False,
        "active_rows": len(active),
        "beneficial_rows": len(beneficial),
        "harmful_rows": len(harmful),
        "recommendation": {
            "decision": decision,
            "online_mode": "shadow",
            "next_evidence": (
                "paired 60-condition calibration plus blinded visual review"
            ),
            "never_modify": [
                "semantic boundary band",
                "stable source structures outside the edit",
                "components above the tiny-island area limit",
            ],
        },
        "rows": deltas,
    }


def _build_contact_sheet(
    rows: list[dict[str, Any]], output_path: Path
) -> Path:
    thumbnails = []
    for row in rows:
        image = Image.open(row["panel"]).convert("RGB")
        image.thumbnail((480, 840))
        canvas = Image.new("RGB", (500, 900), "white")
        canvas.paste(image, ((500 - image.width) // 2, 45))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 10),
            f"{row['organ']} | {row['route_stratum']} | {row['condition_id']}",
            fill="black",
        )
        thumbnails.append(canvas)
    columns = 3
    rows_count = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 500, rows_count * 900), (235, 235, 235))
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * 500, (index // columns) * 900))
    sheet.save(output_path)
    return output_path


def _write_html(rows: list[dict[str, Any]], output_path: Path) -> Path:
    cards = []
    for row in rows:
        metrics = {
            key: row.get(key)
            for key in (
                "status",
                "attempt_count",
                "selected_mode",
                "raw_changed_accuracy",
                "p1_changed_accuracy",
                "raw_changed_macro_miou",
                "p1_changed_macro_miou",
                "raw_drift_u_far",
                "p1_drift_u_far",
                "p1_changed_pixels",
                "p1_operation_count",
            )
        }
        cards.append(
            "<section>"
            f"<h2>{html.escape(row['condition_id'])} · "
            f"{html.escape(row['organ'])} · {html.escape(row['route_stratum'])}</h2>"
            f"<p>{html.escape(row['scenario'])}</p>"
            f"<img loading='lazy' src='{html.escape(row['panel_relative'])}'>"
            f"<pre>{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>"
            "</section>"
        )
    output_path.write_text(
        """<!doctype html><meta charset="utf-8">
<title>Online Agent 18-Condition Canary</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f7;color:#17191c}
section{background:white;border:1px solid #d7dadd;border-radius:6px;margin:0 0 28px;padding:16px}
img{display:block;width:100%;height:auto;border:1px solid #bbb}
pre{white-space:pre-wrap;background:#f2f3f4;padding:12px;overflow:auto}
h1,h2{letter-spacing:0}
</style>
<h1>Online Agent 18-Condition Canary</h1>
<p>G2 validates the frozen online audit loop; it does not alter product behavior.</p>
"""
        + "\n".join(cards),
        encoding="utf-8",
    )
    return output_path


def _grid(
    tiles: list[tuple[str, Image.Image]], *, header: str
) -> Image.Image:
    tile_height = TILE_SIZE + 28
    rows = (len(tiles) + PANEL_COLUMNS - 1) // PANEL_COLUMNS
    canvas = Image.new(
        "RGB",
        (PANEL_COLUMNS * TILE_SIZE, 48 + rows * tile_height),
        (242, 243, 244),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), header, fill=(20, 20, 20), font=ImageFont.load_default())
    for index, (title, image) in enumerate(tiles):
        x = (index % PANEL_COLUMNS) * TILE_SIZE
        y = 48 + (index // PANEL_COLUMNS) * tile_height
        tile = image.convert("RGB").resize((TILE_SIZE, TILE_SIZE), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y))
        draw.text((x + 6, y + TILE_SIZE + 7), title, fill=(20, 20, 20))
    return canvas


def _rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def _binary(path: str | Path) -> np.ndarray:
    return _mask(path) > 0


def _mask_or_none(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    return _mask(path) if path.is_file() else None


def _array_or_none(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    return np.load(path) if path.is_file() else None


def _probabilities_or_none(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as payload:
        return np.asarray(payload["probabilities"])


def _max_probability(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else np.max(values, axis=0)


def _colorize(mask: np.ndarray) -> Image.Image:
    values = np.asarray(mask)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    for class_id, color in UNIFIED_COLOR_MAP.items():
        output[values == class_id] = np.asarray(color, dtype=np.uint8)
    output[values == 255] = (255, 255, 255)
    return Image.fromarray(output)


def _colorize_or_placeholder(mask: np.ndarray | None) -> Image.Image:
    return _placeholder("not applicable") if mask is None else _colorize(mask)


def _heatmap_or_placeholder(values: np.ndarray | None) -> Image.Image:
    if values is None:
        return _placeholder("not applicable")
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
    values = np.clip(values, 0.0, 1.0)
    red = (255 * values).astype(np.uint8)
    blue = (255 * (1.0 - values)).astype(np.uint8)
    green = (255 * (1.0 - np.abs(values - 0.5) * 2.0)).astype(np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=-1))


def _overlay(
    image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    region = np.asarray(mask, dtype=bool)
    base[region] = 0.35 * base[region] + 0.65 * np.asarray(color)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _nuclei_or_placeholder(mask: np.ndarray | None) -> Image.Image:
    if mask is None:
        return _placeholder("not applicable")
    values = np.asarray(mask)
    occupied = values > 0
    output = np.full((*values.shape, 3), 245, dtype=np.uint8)
    output[occupied] = (80, 40, 150)
    return Image.fromarray(output)


def _placeholder(text: str) -> Image.Image:
    image = Image.new("RGB", (512, 512), (225, 227, 229))
    ImageDraw.Draw(image).text((20, 240), text, fill=(70, 70, 70))
    return image


def _json_or_empty(path: Path) -> dict[str, Any]:
    return (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    )


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


__all__ = ["build_online_audit_deck"]

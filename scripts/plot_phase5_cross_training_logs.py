"""Plot Phase 5 cross-training logs with conditional zero losses filtered.

The training loop logs masked losses with ``_masked_mean_or_zero``. When a
micro-batch has no samples for a mode, the corresponding loss is logged as
zero. TensorBoard smoothing can turn those placeholder zeros into misleading
trends. This script filters conditional losses by their matching sample-count
tag before plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


CONDITIONAL_LOSS_SAMPLE_TAGS = {
    "cross_denoise_loss": "cross_samples",
    "counterfactual_denoise_loss": "counterfactual_samples",
    "self_reconstruction_denoise_loss": "self_reconstruction_samples",
}
DEFAULT_LOSS_TAGS = [
    "loss",
    "denoise_loss",
    "cross_denoise_loss",
    "counterfactual_denoise_loss",
    "self_reconstruction_denoise_loss",
]
V2_1_REFERENCE_LOSS_TAGS = [
    "cross_denoise_loss",
    "self_reconstruction_denoise_loss",
    "reference_region_loss_weighted",
    "reference_perceptual_loss_weighted",
]
V2_1_REFERENCE_SKIP_ZERO_LOSS_TAGS = [
    "reference_region_loss_weighted",
    "reference_perceptual_loss_weighted",
]
V2_1_REFERENCE_LOSS_YLIMS = {
    "cross_denoise_loss": (0.4, 0.6),
    "self_reconstruction_denoise_loss": (0.4, 0.6),
    "reference_region_loss_weighted": (0.0, 0.008),
    "reference_perceptual_loss_weighted": (0.0, 0.03),
}
V2_1_REFERENCE_GRAD_TAGS = [
    "reference_grad_ratio",
    "reference_grad_denoise_norm",
    "reference_grad_appearance_norm",
]
V2_1_REFERENCE_GRAD_MEASURED_TAG = "reference_grad_ratio_measured"
DEFAULT_SAMPLE_TAGS = [
    "cross_samples",
    "counterfactual_samples",
    "self_reconstruction_samples",
]


@dataclass(frozen=True)
class ScalarRow:
    tag: str
    step: int
    value: float
    wall_time: float | None = None
    source: str = ""


@dataclass(frozen=True)
class FilteredPoint:
    tag: str
    step: int
    value: float
    rolling: float
    source: str
    valid_reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Phase 5 cross-training scalars while filtering placeholder "
            "zero losses for empty conditional sample groups."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="TensorBoard event file/dir, CSV, or JSONL log path.",
    )
    parser.add_argument("--output-dir", default="training_log_plots")
    parser.add_argument(
        "--preset",
        choices=["default", "cross_v2_1_reference_losses"],
        default="default",
        help=(
            "Use cross_v2_1_reference_losses to plot cross/self denoise plus "
            "weighted region/perceptual reference losses after step 500."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling mean window over valid points only.",
    )
    parser.add_argument(
        "--loss-tags",
        default=None,
        help=(
            "Comma-separated loss scalar tags to plot. Defaults to the selected "
            "preset's tags."
        ),
    )
    parser.add_argument(
        "--after-step",
        type=int,
        default=None,
        help="Only keep scalar points with step greater than this value.",
    )
    parser.add_argument(
        "--skip-zero-loss-tags",
        default=None,
        help=(
            "Comma-separated loss tags whose zero-valued points should be "
            "dropped before plotting."
        ),
    )
    parser.add_argument(
        "--loss-ylim",
        default=None,
        help="Optional y-axis limits for loss plots, formatted as min,max.",
    )
    parser.add_argument(
        "--grad-tags",
        default=None,
        help=(
            "Comma-separated gradient diagnostic scalar tags to plot. Defaults "
            "to the selected preset's gradient tags."
        ),
    )
    parser.add_argument(
        "--grad-measured-tag",
        default=None,
        help=(
            "Optional scalar tag used to filter gradient diagnostics. Points "
            "are kept only when this tag is > 0 at the same step."
        ),
    )
    parser.add_argument(
        "--grad-ylim",
        default=None,
        help="Optional y-axis limits for gradient plots, formatted as min,max.",
    )
    parser.add_argument(
        "--sample-tags",
        default=",".join(DEFAULT_SAMPLE_TAGS),
        help="Comma-separated sample-count tags to plot.",
    )
    parser.add_argument(
        "--keep-zero-losses",
        action="store_true",
        help=(
            "When a conditional loss has no matching sample-count tag, keep "
            "zero-valued points instead of treating them as placeholders."
        ),
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Only draw rolling curves, not faint raw points.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_tag_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_ylim(value: str | None) -> tuple[float, float] | None:
    if value is None or not value.strip():
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"--loss-ylim must be formatted as min,max, got {value!r}.")
    low = float(parts[0])
    high = float(parts[1])
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"--loss-ylim requires finite min < max, got {value!r}.")
    return low, high


def auto_ylim_for_points(points: Iterable[FilteredPoint]) -> tuple[float, float] | None:
    values = [
        value
        for point in points
        for value in (point.value, point.rolling)
        if math.isfinite(value)
    ]
    if not values:
        return None
    low = min(values)
    high = max(values)
    if low == high:
        center = low
        pad = max(abs(center) * 0.05, 1e-6)
        return center - pad, center + pad
    pad = max((high - low) * 0.08, 1e-9)
    return low - pad, high + pad


def auto_ylim_for_points_by_tag(points_by_tag: dict[str, list[FilteredPoint]]) -> tuple[float, float] | None:
    return auto_ylim_for_points(
        point
        for points in points_by_tag.values()
        for point in points
    )


def resolve_ylim_for_tag(
    tag: str,
    *,
    global_ylim: tuple[float, float] | None,
    ylims_by_tag: dict[str, tuple[float, float]] | None,
) -> tuple[float, float] | None:
    if global_ylim is not None:
        return global_ylim
    if ylims_by_tag is None:
        return None
    return ylims_by_tag.get(tag)


def discover_log_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and _is_supported_log_file(candidate)
                )
            )
        elif path.is_file() and _is_supported_log_file(path):
            files.append(path)
        else:
            raise FileNotFoundError(f"Unsupported or missing log path: {path}")
    return files


def _is_supported_log_file(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("events.out.tfevents")
        or path.suffix.lower() in {".csv", ".jsonl", ".json"}
    )


def read_scalar_rows(paths: Iterable[str | Path]) -> list[ScalarRow]:
    rows: list[ScalarRow] = []
    for path in discover_log_files(paths):
        if path.name.startswith("events.out.tfevents"):
            rows.extend(read_tensorboard_event_scalars(path))
        elif path.suffix.lower() == ".csv":
            rows.extend(read_csv_scalars(path))
        elif path.suffix.lower() in {".jsonl", ".json"}:
            rows.extend(read_json_scalars(path))
    return sorted(rows, key=lambda row: (row.tag, row.step, row.source))


def read_tensorboard_event_scalars(path: Path) -> list[ScalarRow]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise RuntimeError(
            "Reading TensorBoard event files requires tensorboard. Install it "
            "in this environment, or export the scalars as CSV/JSONL."
        ) from exc

    accumulator = event_accumulator.EventAccumulator(
        str(path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    rows: list[ScalarRow] = []
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            rows.append(
                ScalarRow(
                    tag=normalize_tag(tag),
                    step=int(event.step),
                    value=float(event.value),
                    wall_time=float(event.wall_time),
                    source=str(path),
                )
            )
    return rows


def read_csv_scalars(path: Path) -> list[ScalarRow]:
    with path.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        return []

    by_lower = {name.lower().replace(" ", "_"): name for name in fieldnames}
    step_col = by_lower.get("step") or by_lower.get("global_step")
    value_col = by_lower.get("value")
    tag_col = by_lower.get("tag") or by_lower.get("metric") or by_lower.get("name")
    wall_col = by_lower.get("wall_time") or by_lower.get("walltime") or by_lower.get("time")

    scalars: list[ScalarRow] = []
    if step_col and value_col:
        inferred_tag = infer_tag_from_filename(path)
        for row in rows:
            tag = normalize_tag(row[tag_col]) if tag_col else inferred_tag
            scalar = parse_scalar_row(row, step_col=step_col, value_col=value_col, wall_col=wall_col, tag=tag, source=path)
            if scalar is not None:
                scalars.append(scalar)
        return scalars

    if not step_col:
        return []

    metric_cols = [
        name
        for name in fieldnames
        if name != step_col and name != wall_col and _looks_numeric_column(rows, name)
    ]
    for row in rows:
        for metric_col in metric_cols:
            scalar = parse_scalar_row(
                row,
                step_col=step_col,
                value_col=metric_col,
                wall_col=wall_col,
                tag=normalize_tag(metric_col),
                source=path,
            )
            if scalar is not None:
                scalars.append(scalar)
    return scalars


def read_json_scalars(path: Path) -> list[ScalarRow]:
    payloads: list[dict] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf8"))
        if isinstance(data, list):
            payloads = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            payloads = [data]
    else:
        with path.open("r", encoding="utf8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        payloads.append(payload)

    scalars: list[ScalarRow] = []
    for payload in payloads:
        step = payload.get("step", payload.get("global_step"))
        if step is None:
            continue
        wall_time = _to_float_or_none(payload.get("wall_time", payload.get("time")))
        for key, value in payload.items():
            if key in {"step", "global_step", "wall_time", "time"}:
                continue
            numeric = _to_float_or_none(value)
            if numeric is None:
                continue
            scalars.append(
                ScalarRow(
                    tag=normalize_tag(key),
                    step=int(step),
                    value=numeric,
                    wall_time=wall_time,
                    source=str(path),
                )
            )
    return scalars


def parse_scalar_row(
    row: dict[str, str],
    *,
    step_col: str,
    value_col: str,
    wall_col: str | None,
    tag: str,
    source: Path,
) -> ScalarRow | None:
    step = _to_float_or_none(row.get(step_col))
    value = _to_float_or_none(row.get(value_col))
    if step is None or value is None:
        return None
    return ScalarRow(
        tag=tag,
        step=int(step),
        value=value,
        wall_time=_to_float_or_none(row.get(wall_col)) if wall_col else None,
        source=str(source),
    )


def _looks_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    for row in rows[:20]:
        value = row.get(column)
        if value not in (None, ""):
            return _to_float_or_none(value) is not None
    return False


def _to_float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def infer_tag_from_filename(path: Path) -> str:
    stem = unquote(path.stem)
    for marker in ("tag-", "tag_", "tag="):
        if marker in stem:
            stem = stem.split(marker, 1)[1]
            break
    return normalize_tag(stem)


def normalize_tag(tag: str) -> str:
    value = str(tag).strip()
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value


def group_by_tag(rows: Iterable[ScalarRow]) -> dict[str, list[ScalarRow]]:
    grouped: dict[str, list[ScalarRow]] = {}
    for row in rows:
        grouped.setdefault(row.tag, []).append(row)
    for tag_rows in grouped.values():
        tag_rows.sort(key=lambda row: row.step)
    return grouped


def filter_loss_points(
    grouped: dict[str, list[ScalarRow]],
    tag: str,
    *,
    rolling_window: int,
    keep_zero_losses: bool = False,
    after_step: int | None = None,
    skip_zero_values: bool = False,
) -> list[FilteredPoint]:
    rows = grouped.get(tag, [])
    sample_tag = CONDITIONAL_LOSS_SAMPLE_TAGS.get(tag)
    sample_by_step = {
        row.step: row.value
        for row in grouped.get(sample_tag or "", [])
    }
    filtered_values: list[float] = []
    points: list[FilteredPoint] = []

    for row in rows:
        if after_step is not None and row.step <= after_step:
            continue
        if skip_zero_values and row.value == 0.0:
            continue
        valid = True
        reason = "unconditional"
        if sample_tag:
            if sample_by_step:
                valid = sample_by_step.get(row.step, 0.0) > 0.0
                reason = f"{sample_tag}>0"
            elif not keep_zero_losses:
                valid = row.value != 0.0
                reason = "value!=0 fallback"
            else:
                reason = "kept zero fallback"
        if not valid:
            continue
        filtered_values.append(row.value)
        points.append(
            FilteredPoint(
                tag=tag,
                step=row.step,
                value=row.value,
                rolling=rolling_mean_tail(filtered_values, rolling_window),
                source=row.source,
                valid_reason=reason,
            )
        )
    return points


def rolling_mean_tail(values: list[float], window: int) -> float:
    if window <= 1:
        return values[-1]
    tail = values[-window:]
    return float(sum(tail) / len(tail))


def build_filtered_loss_points(
    rows: Iterable[ScalarRow],
    loss_tags: Iterable[str],
    *,
    rolling_window: int,
    keep_zero_losses: bool = False,
    after_step: int | None = None,
    skip_zero_loss_tags: Iterable[str] = (),
) -> dict[str, list[FilteredPoint]]:
    grouped = group_by_tag(rows)
    skip_zero_loss_tag_set = set(skip_zero_loss_tags)
    return {
        tag: filter_loss_points(
            grouped,
            tag,
            rolling_window=rolling_window,
            keep_zero_losses=keep_zero_losses,
            after_step=after_step,
            skip_zero_values=tag in skip_zero_loss_tag_set,
        )
        for tag in loss_tags
        if tag in grouped
    }


def build_filtered_scalar_points(
    rows: Iterable[ScalarRow],
    scalar_tags: Iterable[str],
    *,
    rolling_window: int,
    after_step: int | None = None,
    measured_tag: str | None = None,
) -> dict[str, list[FilteredPoint]]:
    grouped = group_by_tag(rows)
    measured_by_step = _measured_steps(grouped.get(measured_tag or "", []))
    points_by_tag: dict[str, list[FilteredPoint]] = {}
    for tag in scalar_tags:
        if tag not in grouped:
            continue
        values: list[float] = []
        points: list[FilteredPoint] = []
        for row in _last_row_by_step(grouped[tag]):
            if after_step is not None and row.step <= after_step:
                continue
            if measured_tag is not None and measured_by_step:
                if row.step not in measured_by_step:
                    continue
                reason = f"{measured_tag}>0"
            elif measured_tag is not None:
                if row.value == 0.0:
                    continue
                reason = "value!=0 fallback"
            else:
                reason = "unconditional"
            values.append(row.value)
            points.append(
                FilteredPoint(
                    tag=tag,
                    step=row.step,
                    value=row.value,
                    rolling=rolling_mean_tail(values, rolling_window),
                    source=row.source,
                    valid_reason=reason,
                )
            )
        points_by_tag[tag] = points
    return points_by_tag


def _last_row_by_step(rows: list[ScalarRow]) -> list[ScalarRow]:
    """Collapse duplicate TensorBoard events emitted at the same global step."""
    by_step: dict[int, ScalarRow] = {}
    for row in sorted(rows, key=lambda item: (item.step, item.wall_time if item.wall_time is not None else -math.inf)):
        by_step[row.step] = row
    return [by_step[step] for step in sorted(by_step)]


def _measured_steps(rows: list[ScalarRow]) -> set[int]:
    return {row.step for row in rows if row.value > 0.0}


def write_filtered_points_csv(path: Path, points_by_tag: dict[str, list[FilteredPoint]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tag", "step", "value", "rolling", "valid_reason", "source"],
        )
        writer.writeheader()
        for tag in sorted(points_by_tag):
            for point in points_by_tag[tag]:
                writer.writerow(
                    {
                        "tag": point.tag,
                        "step": point.step,
                        "value": point.value,
                        "rolling": point.rolling,
                        "valid_reason": point.valid_reason,
                        "source": point.source,
                    }
                )


def plot_losses(
    output_path: Path,
    points_by_tag: dict[str, list[FilteredPoint]],
    *,
    rolling_window: int,
    draw_raw: bool,
    ylim: tuple[float, float] | None,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for tag, points in points_by_tag.items():
        if not points:
            continue
        steps = [point.step for point in points]
        values = [point.value for point in points]
        rolling = [point.rolling for point in points]
        if draw_raw:
            ax.plot(steps, values, alpha=0.18, linewidth=0.8)
        ax.plot(steps, rolling, linewidth=1.8, label=f"{tag} (valid rolling {rolling_window})")

    ax.set_title("Filtered conditional losses")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    effective_ylim = ylim if ylim is not None else auto_ylim_for_points_by_tag(points_by_tag)
    if effective_ylim is not None:
        ax.set_ylim(*effective_ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_scalar_points(
    output_path: Path,
    points: list[FilteredPoint],
    *,
    rolling_window: int,
    draw_raw: bool,
    ylim: tuple[float, float] | None,
    dpi: int,
    title: str,
    ylabel: str,
) -> None:
    if not points:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    steps = [point.step for point in points]
    values = [point.value for point in points]
    rolling = [point.rolling for point in points]
    if draw_raw:
        ax.plot(steps, values, alpha=0.18, linewidth=0.8)
    ax.plot(steps, rolling, linewidth=1.8, label=f"rolling {rolling_window}")

    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel(ylabel)
    effective_ylim = ylim if ylim is not None else auto_ylim_for_points(points)
    if effective_ylim is not None:
        ax.set_ylim(*effective_ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_scalar_points_by_tag(
    output_dir: Path,
    points_by_tag: dict[str, list[FilteredPoint]],
    *,
    rolling_window: int,
    draw_raw: bool,
    ylim: tuple[float, float] | None,
    ylims_by_tag: dict[str, tuple[float, float]] | None = None,
    dpi: int,
    ylabel: str,
    title_prefix: str,
) -> None:
    for tag, points in points_by_tag.items():
        plot_scalar_points(
            output_dir / f"{safe_filename(tag)}.png",
            points,
            rolling_window=rolling_window,
            draw_raw=draw_raw,
            ylim=resolve_ylim_for_tag(
                tag,
                global_ylim=ylim,
                ylims_by_tag=ylims_by_tag,
            ),
            dpi=dpi,
            title=f"{title_prefix}: {tag}",
            ylabel=ylabel,
        )


def plot_samples(
    output_path: Path,
    grouped: dict[str, list[ScalarRow]],
    sample_tags: Iterable[str],
    *,
    rolling_window: int,
    draw_raw: bool,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    for tag in sample_tags:
        rows = grouped.get(tag, [])
        if not rows:
            continue
        values: list[float] = []
        steps: list[int] = []
        rolling: list[float] = []
        for row in rows:
            values.append(row.value)
            steps.append(row.step)
            rolling.append(rolling_mean_tail(values, rolling_window))
        if draw_raw:
            ax.plot(steps, values, alpha=0.15, linewidth=0.8)
        ax.plot(steps, rolling, linewidth=1.8, label=f"{tag} (rolling {rolling_window})")

    ax.set_title("Sample mix")
    ax.set_xlabel("global step")
    ax.set_ylabel("samples per logged micro-batch")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def safe_filename(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "scalar"


def write_summary(
    path: Path,
    rows: list[ScalarRow],
    points_by_tag: dict[str, list[FilteredPoint]],
    *,
    after_step: int | None,
    skip_zero_loss_tags: Iterable[str],
    grad_points_by_tag: dict[str, list[FilteredPoint]] | None = None,
    grad_measured_tag: str | None = None,
) -> None:
    grouped = group_by_tag(rows)
    summary = {
        "num_scalar_rows": len(rows),
        "available_tags": sorted(grouped),
        "after_step": after_step,
        "skip_zero_loss_tags": sorted(set(skip_zero_loss_tags)),
        "filtered_loss_points": {
            tag: len(points)
            for tag, points in sorted(points_by_tag.items())
        },
        "filtered_grad_points": {
            tag: len(points)
            for tag, points in sorted((grad_points_by_tag or {}).items())
        },
        "grad_measured_tag": grad_measured_tag,
        "conditional_loss_sample_tags": CONDITIONAL_LOSS_SAMPLE_TAGS,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    rows = read_scalar_rows(args.input)
    if not rows:
        raise RuntimeError("No scalar rows found in the provided logs.")

    if args.preset == "cross_v2_1_reference_losses":
        default_loss_tags = V2_1_REFERENCE_LOSS_TAGS
        default_skip_zero_loss_tags = V2_1_REFERENCE_SKIP_ZERO_LOSS_TAGS
        default_grad_tags = V2_1_REFERENCE_GRAD_TAGS
        default_grad_measured_tag = V2_1_REFERENCE_GRAD_MEASURED_TAG
        default_after_step = 500
        default_loss_ylims_by_tag = V2_1_REFERENCE_LOSS_YLIMS
    else:
        default_loss_tags = DEFAULT_LOSS_TAGS
        default_skip_zero_loss_tags = []
        default_grad_tags = []
        default_grad_measured_tag = None
        default_after_step = None
        default_loss_ylims_by_tag = {}

    loss_tags = parse_tag_list(args.loss_tags) if args.loss_tags is not None else list(default_loss_tags)
    grad_tags = parse_tag_list(args.grad_tags) if args.grad_tags is not None else list(default_grad_tags)
    grad_measured_tag = args.grad_measured_tag if args.grad_measured_tag is not None else default_grad_measured_tag
    skip_zero_loss_tags = (
        parse_tag_list(args.skip_zero_loss_tags)
        if args.skip_zero_loss_tags is not None
        else list(default_skip_zero_loss_tags)
    )
    after_step = args.after_step if args.after_step is not None else default_after_step
    sample_tags = parse_tag_list(args.sample_tags)
    loss_ylim = parse_ylim(args.loss_ylim)
    grad_ylim = parse_ylim(args.grad_ylim)
    grouped = group_by_tag(rows)
    points_by_tag = build_filtered_loss_points(
        rows,
        loss_tags,
        rolling_window=max(1, args.rolling_window),
        keep_zero_losses=args.keep_zero_losses,
        after_step=after_step,
        skip_zero_loss_tags=skip_zero_loss_tags,
    )
    grad_points_by_tag = build_filtered_scalar_points(
        rows,
        grad_tags,
        rolling_window=max(1, args.rolling_window),
        after_step=after_step,
        measured_tag=grad_measured_tag if grad_tags else None,
    )
    write_filtered_points_csv(output_dir / "filtered_loss_scalars.csv", points_by_tag)
    write_filtered_points_csv(output_dir / "filtered_grad_scalars.csv", grad_points_by_tag)
    write_summary(
        output_dir / "summary.json",
        rows,
        points_by_tag,
        after_step=after_step,
        skip_zero_loss_tags=skip_zero_loss_tags,
        grad_points_by_tag=grad_points_by_tag,
        grad_measured_tag=grad_measured_tag if grad_tags else None,
    )
    plot_losses(
        output_dir / "filtered_losses.png",
        points_by_tag,
        rolling_window=max(1, args.rolling_window),
        draw_raw=not args.no_raw,
        ylim=loss_ylim,
        dpi=args.dpi,
    )
    plot_scalar_points_by_tag(
        output_dir / "losses",
        points_by_tag,
        rolling_window=max(1, args.rolling_window),
        draw_raw=not args.no_raw,
        ylim=loss_ylim,
        ylims_by_tag=default_loss_ylims_by_tag,
        dpi=args.dpi,
        ylabel="loss",
        title_prefix="Filtered loss",
    )
    plot_scalar_points_by_tag(
        output_dir / "gradients",
        grad_points_by_tag,
        rolling_window=max(1, args.rolling_window),
        draw_raw=not args.no_raw,
        ylim=grad_ylim,
        ylims_by_tag=None,
        dpi=args.dpi,
        ylabel="gradient diagnostic",
        title_prefix="Reference gradient diagnostic",
    )
    plot_samples(
        output_dir / "sample_mix.png",
        grouped,
        sample_tags,
        rolling_window=max(1, args.rolling_window),
        draw_raw=not args.no_raw,
        dpi=args.dpi,
    )
    print(f"wrote plots to {output_dir}")
    print(f"available tags: {', '.join(sorted(grouped))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

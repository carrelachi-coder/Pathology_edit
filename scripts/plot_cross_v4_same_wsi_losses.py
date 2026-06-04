"""Plot Cross V4 same-WSI perceptual/style/denoise losses together.

Reads TensorBoard event files, CSV, JSON, or JSONL logs. The same-WSI
perceptual loss is computed on an interval and logged as zero on skipped steps,
so this script filters it with ``same_wsi_perceptual_layers > 0`` before
plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from plot_phase5_cross_training_logs import (
    FilteredPoint,
    ScalarRow,
    discover_log_files,
    group_by_tag,
    parse_ylim,
    read_scalar_rows,
    rolling_mean_tail,
)

DEFAULT_TAGS = ("denoise_loss", "style_loss", "same_wsi_perceptual_loss")
DEFAULT_AUX_TAGS = ("style_loss", "same_wsi_perceptual_loss")
GATE_TAGS = {
    "same_wsi_perceptual_loss": "same_wsi_perceptual_layers",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="TensorBoard event file/dir, CSV, JSON, or JSONL log path.",
    )
    parser.add_argument(
        "--output-dir",
        default="cross_v4_same_wsi_loss_plots",
        help="Directory for PNG/CSV/summary outputs.",
    )
    parser.add_argument(
        "--tags",
        default=",".join(DEFAULT_TAGS),
        help="Comma-separated scalar tags to draw on the combined plot.",
    )
    parser.add_argument(
        "--since-log",
        default=None,
        help=(
            "Only read TensorBoard/CSV/JSON files modified at or after this log file. "
            "Use the nohup log for the run to avoid mixing old event files."
        ),
    )
    parser.add_argument(
        "--since-manifest",
        default=None,
        help="Only read files modified at or after this nohup manifest JSON's started_at time.",
    )
    parser.add_argument(
        "--mtime-window-seconds",
        type=float,
        default=300.0,
        help="When using --since-log, include files up to this many seconds older than the log mtime.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling mean window over valid points.",
    )
    parser.add_argument("--ylim", default=None, help="Optional y-axis limits formatted as min,max.")
    parser.add_argument(
        "--aux-tags",
        default=",".join(DEFAULT_AUX_TAGS),
        help="Comma-separated tags for a second zoomed auxiliary-loss plot.",
    )
    parser.add_argument(
        "--aux-ylim",
        default=None,
        help="Optional y-axis limits for the auxiliary-loss zoom plot, formatted as min,max.",
    )
    parser.add_argument("--no-raw", action="store_true", help="Only draw rolling curves.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    input_paths = _filter_inputs_by_mtime(
        args.input,
        since_log=args.since_log,
        since_manifest=args.since_manifest,
        mtime_window_seconds=args.mtime_window_seconds,
    )
    rows = read_scalar_rows(input_paths)
    if not rows:
        raise RuntimeError("No scalar rows found in the provided logs.")
    grouped = group_by_tag(rows)
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    points_by_tag = {
        tag: _valid_points_for_tag(
            grouped,
            tag,
            rolling_window=max(1, args.rolling_window),
        )
        for tag in tags
        if tag in grouped
    }
    if not any(points_by_tag.values()):
        raise RuntimeError(
            f"No requested tags found. Requested={tags}; available={sorted(grouped)}"
        )

    _write_points_csv(output_dir / "same_wsi_style_denoise_losses.csv", points_by_tag)
    _write_summary(output_dir / "summary.json", grouped, points_by_tag, input_paths=input_paths)
    _plot_combined(
        output_dir / "same_wsi_style_denoise_losses.png",
        points_by_tag,
        rolling_window=max(1, args.rolling_window),
        draw_raw=not args.no_raw,
        ylim=parse_ylim(args.ylim),
        dpi=args.dpi,
    )
    aux_tags = [tag.strip() for tag in args.aux_tags.split(",") if tag.strip()]
    aux_points_by_tag = {
        tag: points_by_tag[tag]
        for tag in aux_tags
        if tag in points_by_tag and points_by_tag[tag]
    }
    if aux_points_by_tag:
        _plot_combined(
            output_dir / "same_wsi_style_losses_zoom.png",
            aux_points_by_tag,
            rolling_window=max(1, args.rolling_window),
            draw_raw=not args.no_raw,
            ylim=parse_ylim(args.aux_ylim),
            dpi=args.dpi,
            title="Cross V4 Auxiliary Losses: Same-WSI Perceptual / Style",
        )
    print(f"wrote plot: {output_dir / 'same_wsi_style_denoise_losses.png'}")
    if aux_points_by_tag:
        print(f"wrote zoom plot: {output_dir / 'same_wsi_style_losses_zoom.png'}")
    print(f"wrote csv: {output_dir / 'same_wsi_style_denoise_losses.csv'}")
    print(f"read files: {len(input_paths)}")
    for path in input_paths[:10]:
        print(f"  {path}")
    if len(input_paths) > 10:
        print(f"  ... {len(input_paths) - 10} more")
    print(f"available tags: {', '.join(sorted(grouped))}")
    return 0


def _valid_points_for_tag(
    grouped: dict[str, list[ScalarRow]],
    tag: str,
    *,
    rolling_window: int,
) -> list[FilteredPoint]:
    rows = grouped.get(tag, [])
    gate_tag = GATE_TAGS.get(tag)
    gate_by_step = {row.step: row.value for row in grouped.get(gate_tag or "", [])}
    values: list[float] = []
    points: list[FilteredPoint] = []
    for row in rows:
        if gate_tag and gate_by_step:
            if gate_by_step.get(row.step, 0.0) <= 0.0:
                continue
            reason = f"{gate_tag}>0"
        elif gate_tag:
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
    return points


def _plot_combined(
    output_path: Path,
    points_by_tag: dict[str, list[FilteredPoint]],
    *,
    rolling_window: int,
    draw_raw: bool,
    ylim: tuple[float, float] | None,
    dpi: int,
    title: str = "Cross V4 Losses: Same-WSI Perceptual / Style / Denoise",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for tag, points in points_by_tag.items():
        if not points:
            continue
        steps = [point.step for point in points]
        values = [point.value for point in points]
        rolling = [point.rolling for point in points]
        if draw_raw:
            ax.plot(steps, values, alpha=0.16, linewidth=0.8)
        ax.plot(steps, rolling, linewidth=1.9, label=f"{tag} (rolling {rolling_window})")
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _write_points_csv(path: Path, points_by_tag: dict[str, list[FilteredPoint]]) -> None:
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


def _write_summary(
    path: Path,
    grouped: dict[str, list[ScalarRow]],
    points_by_tag: dict[str, list[FilteredPoint]],
    *,
    input_paths: list[Path],
) -> None:
    summary = {
        "input_files": [str(path) for path in input_paths],
        "available_tags": sorted(grouped),
        "requested_points": {tag: len(points) for tag, points in sorted(points_by_tag.items())},
        "gate_tags": GATE_TAGS,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")


def _filter_inputs_by_mtime(
    inputs: list[str],
    *,
    since_log: str | None,
    since_manifest: str | None,
    mtime_window_seconds: float,
) -> list[Path]:
    files = discover_log_files(inputs)
    cutoff = None
    if since_log:
        cutoff = Path(since_log).stat().st_mtime - max(0.0, float(mtime_window_seconds))
    if since_manifest:
        manifest_cutoff = _manifest_started_at_timestamp(Path(since_manifest))
        cutoff = manifest_cutoff if cutoff is None else max(cutoff, manifest_cutoff)
    if cutoff is None:
        return files
    filtered = [path for path in files if path.stat().st_mtime >= cutoff]
    if not filtered:
        raise RuntimeError(
            f"No log/event files survived mtime filtering. cutoff={cutoff}, inputs={inputs}"
        )
    return filtered


def _manifest_started_at_timestamp(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf8"))
    started_at = payload.get("started_at")
    if not started_at:
        return path.stat().st_mtime
    value = str(started_at).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        from datetime import datetime

        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return path.stat().st_mtime


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross V1 training health sentinel.

The script parses train logs and optionally runs fixed feature probes for
projected/encoder_hid_proj tumor similarity. It is intentionally read-only:
no checkpoints are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REFERENCE_HEALTH_RE = re.compile(
    r"Reference health step=(?P<step>\d+) variant=(?P<variant>[\w.]+): "
    r"pred_l2=(?P<pred_l2>[-+0-9.eE]+) "
    r"loss_gap=(?P<loss_gap>[-+0-9.eE]+)"
    r"(?:[±+/-]+(?P<stderr>[-+0-9.eE]+) n=(?P<n>\d+))? "
    r"first_double_ip_output_cos=(?P<cos>[-+0-9.eE]+)"
)

IP_RATIO_RE = re.compile(
    r"IP attention health step=(?P<step>\d+) variant=(?P<variant>[\w.]+): "
    r"blocks=(?P<blocks>\d+) ratio\[min/mean/max\]="
    r"(?P<ratio_min>[-+0-9.eE]+)/(?P<ratio_mean>[-+0-9.eE]+)/(?P<ratio_max>[-+0-9.eE]+)"
)

REGION_MASK_RE = re.compile(
    r"IP attention health step=(?P<step>\d+) variant=(?P<variant>[\w.]+) region_mask: "
    r".*?allowed_pairs=(?P<allowed_pairs>[-+0-9.eE]+) "
    r"active_q=(?P<active_q>[-+0-9.eE]+) "
    r"missing_q=(?P<missing_q>[-+0-9.eE]+) "
    r"fallback_q=(?P<fallback_q>[-+0-9.eE]+) "
    r"null_q=(?P<null_q>[-+0-9.eE]+)"
)

PARAM_DELTA_RE = re.compile(
    r"IP/ref param delta health (?P<group>[\w.]+): "
    r"tensors=(?P<tensors>\d+) params=(?P<params>\d+) .*?"
    r"delta_norm=(?P<delta_norm>[-+0-9.eE]+) "
    r"relative_delta=(?P<relative_delta>[-+0-9.eE]+) "
    r"max_abs=(?P<max_abs>[-+0-9.eE]+) "
    r"grad_ever_nonzero=(?P<grad>\w+)"
)

PARAM_STEP_RE = re.compile(r"IP/ref param delta health step=(?P<step>\d+)")

TOKEN_DEBUG_RE = re.compile(
    r"Reference token debug (?P<stage>[\w_]+): .*?"
    r"token_norm_mean=(?P<token_norm_mean>[-+0-9.eE]+) .*?"
    r"within_sample_token_std=(?P<within_sample_token_std>[-+0-9.eE]+)"
    r"(?: .*?batch_centered_l2_mean=(?P<centered_l2>[-+0-9.eE]+))?"
)

REFERENCE_SIGNAL_STEP_RE = re.compile(r"Reference signal debug step=(?P<step>\d+)")

SIGNAL_DEBUG_RE = re.compile(
    r"Reference signal debug (?P<stage>[\w_]+): .*?"
    r"finite=(?P<finite>[-+0-9.eE]+)"
)

NUMERIC_ANOMALY_RE = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])", re.IGNORECASE)
STEP_IN_LINE_RE = re.compile(r"step=(?P<step>\d+)")
EMPTY_T_BUCKET_RE = re.compile(
    r"ip_health_[\w.]+_loss_gap_t_(?:low|mid|high)(?:_stderr)?=nan",
    re.IGNORECASE,
)

VARIANT_ORDER = ("normal", "zero", "real_feature", "real_label", "real")
FEATURE_ONLY_VARIANTS = ("zero", "real_feature")


@dataclass
class StepHealth:
    step: int
    reference: dict[str, dict[str, float]] = field(default_factory=dict)
    ratios: dict[str, dict[str, float]] = field(default_factory=dict)
    regions: dict[str, dict[str, float]] = field(default_factory=dict)
    param_delta: dict[str, dict[str, float | bool]] = field(default_factory=dict)
    token_debug: dict[str, dict[str, float]] = field(default_factory=dict)
    signal_debug: dict[str, dict[str, float]] = field(default_factory=dict)
    numeric_anomalies: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Cross V1 regional IP training health.")
    parser.add_argument("--run-dir", required=True, help="CROSS_V1_OUTPUT_DIR.")
    parser.add_argument("--log-file", default=None, help="Defaults to <run-dir>/logs/latest.log.")
    parser.add_argument("--metadata", default="phase5_runs/cross_meta/metadata_cross_train.json")
    parser.add_argument("--uni-checkpoint", default="UNI-2h/pytorch_model.bin")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--steps", default="500,1000,2000,3000,5000")
    parser.add_argument("--feature-probe-every", type=int, default=500)
    parser.add_argument("--run-feature-probes", action="store_true")
    parser.add_argument("--feature-max-samples", type=int, default=1000)
    parser.add_argument("--feature-samples-per-wsi", type=int, default=20)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--margin-warn", type=float, default=0.05)
    parser.add_argument("--margin-ok", type=float, default=0.15)
    parser.add_argument("--ratio-warn", type=float, default=0.5)
    parser.add_argument("--output-json", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    log_file = Path(args.log_file) if args.log_file else run_dir / "logs" / "latest.log"
    project_root = Path(args.project_root)
    requested_steps = parse_step_list(args.steps)

    parsed = parse_training_log(log_file)
    checkpoints = discover_checkpoints(run_dir)

    feature_summaries: dict[str, dict[str, Any]] = {}
    if args.run_feature_probes:
        for step in requested_steps:
            checkpoint = checkpoints.get(step)
            if checkpoint is None or step % max(1, args.feature_probe_every) != 0:
                continue
            feature_summaries[str(step)] = run_feature_probes(
                args=args,
                project_root=project_root,
                checkpoint=checkpoint,
                step=step,
            )

    report = build_report(
        parsed=parsed,
        checkpoints=checkpoints,
        feature_summaries=feature_summaries,
        requested_steps=requested_steps,
        margin_warn=float(args.margin_warn),
        margin_ok=float(args.margin_ok),
        ratio_warn=float(args.ratio_warn),
    )
    text = format_report(report)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    return 0


def parse_step_list(value: str) -> list[int]:
    steps = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            steps.append(int(part))
    return sorted(set(steps))


def parse_training_log(path: Path) -> dict[int, StepHealth]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    steps: dict[int, StepHealth] = {}
    current_param_step: int | None = None
    current_token_step: int | None = None

    for line in path.read_text(encoding="utf8", errors="replace").splitlines():
        anomaly_match = NUMERIC_ANOMALY_RE.search(line)
        if anomaly_match:
            # Inactive optional groups are logged with finite=nan because they
            # have zero tensors. That is a sentinel value, not a numerical
            # failure in the active training path.
            if "trainable_tensors=0" in line or "params=0" in line:
                anomaly_match = None
            # A health probe can contain only one sample, so two of the three
            # timestep buckets are normally empty. Their mean/stderr is logged
            # as NaN with n=0 and must not be treated as a model NaN.
            elif EMPTY_T_BUCKET_RE.search(line):
                anomaly_match = None
        if anomaly_match:
            step_match = STEP_IN_LINE_RE.search(line)
            anomaly_step = (
                int(step_match.group("step"))
                if step_match
                else current_token_step or current_param_step
            )
            if anomaly_step is not None:
                item = steps.setdefault(anomaly_step, StepHealth(step=anomaly_step))
                item.numeric_anomalies.append(line.strip())

        match = REFERENCE_HEALTH_RE.search(line)
        if match:
            step = int(match.group("step"))
            item = steps.setdefault(step, StepHealth(step=step))
            variant = match.group("variant")
            item.reference[variant] = {
                "pred_l2": parse_float(match.group("pred_l2")),
                "loss_gap": parse_float(match.group("loss_gap")),
                "loss_gap_stderr": parse_float(match.group("stderr")),
                "loss_gap_n": parse_float(match.group("n")),
                "first_ip_output_cos": parse_float(match.group("cos")),
            }
            current_token_step = step
            continue

        match = REFERENCE_SIGNAL_STEP_RE.search(line)
        if match:
            current_token_step = int(match.group("step"))
            steps.setdefault(current_token_step, StepHealth(step=current_token_step))
            continue

        match = SIGNAL_DEBUG_RE.search(line)
        if match and current_token_step is not None:
            item = steps.setdefault(current_token_step, StepHealth(step=current_token_step))
            stage = match.group("stage")
            item.signal_debug[stage] = {
                "finite": parse_float(match.group("finite")),
            }
            continue

        match = IP_RATIO_RE.search(line)
        if match:
            step = int(match.group("step"))
            item = steps.setdefault(step, StepHealth(step=step))
            variant = match.group("variant")
            item.ratios[variant] = {
                "blocks": parse_float(match.group("blocks")),
                "ratio_min": parse_float(match.group("ratio_min")),
                "ratio_mean": parse_float(match.group("ratio_mean")),
                "ratio_max": parse_float(match.group("ratio_max")),
            }
            continue

        match = REGION_MASK_RE.search(line)
        if match:
            step = int(match.group("step"))
            item = steps.setdefault(step, StepHealth(step=step))
            variant = match.group("variant")
            item.regions[variant] = {
                "allowed_pairs": parse_float(match.group("allowed_pairs")),
                "active_q": parse_float(match.group("active_q")),
                "missing_q": parse_float(match.group("missing_q")),
                "fallback_q": parse_float(match.group("fallback_q")),
                "null_q": parse_float(match.group("null_q")),
            }
            continue

        match = PARAM_STEP_RE.search(line)
        if match:
            current_param_step = int(match.group("step"))
            steps.setdefault(current_param_step, StepHealth(step=current_param_step))
            continue

        match = PARAM_DELTA_RE.search(line)
        if match and current_param_step is not None:
            item = steps.setdefault(current_param_step, StepHealth(step=current_param_step))
            group = match.group("group")
            item.param_delta[group] = {
                "tensors": parse_float(match.group("tensors")),
                "params": parse_float(match.group("params")),
                "delta_norm": parse_float(match.group("delta_norm")),
                "relative_delta": parse_float(match.group("relative_delta")),
                "max_abs": parse_float(match.group("max_abs")),
                "grad_ever_nonzero": match.group("grad").lower() == "true",
            }
            continue

        match = TOKEN_DEBUG_RE.search(line)
        if match and current_token_step is not None:
            item = steps.setdefault(current_token_step, StepHealth(step=current_token_step))
            stage = match.group("stage")
            item.token_debug[stage] = {
                "token_norm_mean": parse_float(match.group("token_norm_mean")),
                "within_sample_token_std": parse_float(match.group("within_sample_token_std")),
                "batch_centered_l2_mean": parse_float(match.group("centered_l2")),
            }
            continue
    return steps


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def discover_checkpoints(run_dir: Path) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints[step] = path
    return checkpoints


def run_feature_probes(
    *,
    args: argparse.Namespace,
    project_root: Path,
    checkpoint: Path,
    step: int,
) -> dict[str, Any]:
    out_root = Path(args.run_dir) / "diagnostics" / f"checkpoint-{step}"
    summaries: dict[str, Any] = {}
    for stage in ("projected", "encoder_hid_proj"):
        output_dir = out_root / stage
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "analyze_uni_tumor_similarity.py"),
            "--metadata",
            str(project_root / args.metadata),
            "--uni-checkpoint",
            str(project_root / args.uni_checkpoint),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--feature-stage",
            stage,
            "--tumor-label",
            "1",
            "--max-samples",
            str(args.feature_max_samples),
            "--samples-per-wsi",
            str(args.feature_samples_per_wsi),
            "--batch-size",
            str(args.feature_batch_size),
            "--device",
            str(args.device),
            "--torch-dtype",
            str(args.torch_dtype),
        ]
        subprocess.run(cmd, cwd=project_root, check=True)
        summary_path = output_dir / "tumor_similarity_summary.json"
        summaries[stage] = json.loads(summary_path.read_text(encoding="utf8"))
    return summaries


def build_report(
    *,
    parsed: dict[int, StepHealth],
    checkpoints: dict[int, Path],
    feature_summaries: dict[str, dict[str, Any]],
    requested_steps: list[int],
    margin_warn: float,
    margin_ok: float,
    ratio_warn: float,
) -> dict[str, Any]:
    latest_step = max(parsed) if parsed else 0
    latest = parsed.get(latest_step)
    decisions = []
    per_step = {}
    for step in requested_steps:
        nearest_step = max((s for s in parsed if s <= step), default=None)
        if latest_step < step:
            step_report = summarize_missing_future_step(
                target_step=step,
                latest_step=latest_step,
                checkpoint=checkpoints.get(step),
            )
        else:
            item = parsed.get(nearest_step) if nearest_step is not None else None
            step_report = summarize_step(
                item,
                checkpoint=checkpoints.get(step),
                feature_summary=feature_summaries.get(str(step), {}),
                margin_warn=margin_warn,
                margin_ok=margin_ok,
                ratio_warn=ratio_warn,
            )
        per_step[str(step)] = step_report
        decisions.extend(step_report["decisions"])

    gates = {
        "instrument_500_1000": gate_instrument(per_step),
        "early_learning_2000_3000": gate_early_learning(per_step),
        "stop_loss_5000": gate_stop_loss(per_step),
    }
    overall = worst_status([*decisions, *[gate["status"] for gate in gates.values()]])
    return {
        "overall": overall,
        "latest_logged_step": latest_step,
        "latest": summarize_step(
            latest,
            checkpoint=checkpoints.get(latest_step),
            feature_summary=feature_summaries.get(str(latest_step), {}),
            margin_warn=margin_warn,
            margin_ok=margin_ok,
            ratio_warn=ratio_warn,
        ) if latest is not None else None,
        "gates": gates,
        "steps": per_step,
        "feature_probe_steps": sorted(feature_summaries.keys(), key=int),
    }


def summarize_step(
    item: StepHealth | None,
    *,
    checkpoint: Path | None,
    feature_summary: dict[str, Any],
    margin_warn: float,
    margin_ok: float,
    ratio_warn: float,
) -> dict[str, Any]:
    if item is None:
        return {
            "status": "WARN",
            "step": None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "decisions": ["WARN"],
            "notes": ["no health logs found at or before this step"],
        }
    notes = []
    decisions = []
    has_ip_health = bool(item.reference or item.ratios or item.regions or item.param_delta)
    if not has_ip_health:
        return {
            "status": "WARN",
            "step": item.step,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "decisions": ["WARN"],
            "notes": ["no IP health diagnostics found at this logged step yet"],
            "reference": item.reference,
            "ratios": item.ratios,
            "regions": item.regions,
            "region_alignment": build_region_alignment(item),
            "ratio_summary": build_ratio_summary(item),
            "finite_summary": build_finite_summary(item),
            "numeric_anomalies": item.numeric_anomalies[:10],
            "param_delta": item.param_delta,
            "token_debug": item.token_debug,
            "signal_debug": item.signal_debug,
            "feature_margins": {},
        }

    for variant in VARIANT_ORDER:
        ratio = item.ratios.get(variant)
        if ratio and ratio["ratio_max"] > ratio_warn:
            notes.append(f"{variant} ratio_max {ratio['ratio_max']:.3f} > {ratio_warn:.3f}")
            decisions.append("WARN")

    finite_issues = finite_issue_notes(item)
    if finite_issues:
        notes.extend(finite_issues)
        decisions.append("FAIL")

    if item.numeric_anomalies:
        notes.append(f"numeric anomaly line(s) found: {len(item.numeric_anomalies)}")
        decisions.append("FAIL")

    required_reference_variants = ("zero", "real_feature", "real_label", "real")
    missing_reference_variants = [
        variant for variant in required_reference_variants if variant not in item.reference
    ]
    if missing_reference_variants:
        notes.append(
            "missing reference health arm(s): "
            + ", ".join(missing_reference_variants)
            + "; instrument is invalid, likely old training code/process"
        )
        decisions.append("FAIL")

    required_region_variants = ("normal", "zero", "real_feature", "real_label", "real")
    missing_region_variants = [
        variant for variant in required_region_variants if variant not in item.regions
    ]
    if item.regions and missing_region_variants:
        notes.append(
            "missing region-mask health arm(s): "
            + ", ".join(missing_region_variants)
            + "; cannot verify active_q alignment"
        )
        decisions.append("FAIL")

    normal_region = item.regions.get("normal")
    for variant in FEATURE_ONLY_VARIANTS:
        feature_region = item.regions.get(variant)
        if normal_region and feature_region:
            diff = abs(feature_region["active_q"] - normal_region["active_q"])
            if diff > 0.05:
                notes.append(f"{variant} active_q differs from normal by {diff:.3f}")
                decisions.append("FAIL")

    real_label_region = item.regions.get("real_label")
    real_region = item.regions.get("real")
    if real_label_region and real_region:
        diff = abs(real_region["active_q"] - real_label_region["active_q"])
        if diff > 0.05:
            notes.append(f"real active_q differs from real_label by {diff:.3f}")
            decisions.append("FAIL")

    for group, values in item.param_delta.items():
        if values.get("params", math.nan) == 0.0 or values.get("tensors", math.nan) == 0.0:
            continue
        if values.get("delta_norm", math.nan) == 0.0:
            notes.append(f"{group} param delta is zero")
            decisions.append("FAIL")
        if not values.get("grad_ever_nonzero", False):
            notes.append(f"{group} never saw nonzero grad")
            decisions.append("WARN")

    margins = {}
    for stage, summary in feature_summary.items():
        margin = feature_margin(summary)
        margins[stage] = margin
        if math.isfinite(margin):
            if margin < margin_warn:
                notes.append(f"{stage} same-vs-different WSI margin {margin:.3f} < {margin_warn:.3f}")
                decisions.append("FAIL")
            elif margin < margin_ok:
                notes.append(f"{stage} margin {margin:.3f} below ok threshold {margin_ok:.3f}")
                decisions.append("WARN")

    if not decisions:
        decisions.append("PASS")
    return {
        "status": worst_status(decisions),
        "step": item.step,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "decisions": decisions,
        "notes": notes,
        "reference": item.reference,
        "ratios": item.ratios,
        "regions": item.regions,
        "region_alignment": build_region_alignment(item),
        "ratio_summary": build_ratio_summary(item),
        "finite_summary": build_finite_summary(item),
        "numeric_anomalies": item.numeric_anomalies[:10],
        "param_delta": item.param_delta,
        "token_debug": item.token_debug,
        "signal_debug": item.signal_debug,
        "feature_margins": margins,
    }


def summarize_missing_future_step(
    *,
    target_step: int,
    latest_step: int,
    checkpoint: Path | None,
) -> dict[str, Any]:
    note = (
        f"latest health log step {latest_step} is before requested step {target_step}; "
        "not enough training has run for this gate"
    )
    return {
        "status": "WARN",
        "step": latest_step if latest_step > 0 else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "decisions": ["WARN"],
        "notes": [note],
    }


def feature_margin(summary: dict[str, Any]) -> float:
    sample_level = summary.get("sample_level") or {}
    same = ((sample_level.get("same_wsi") or {}).get("mean"))
    diff = ((sample_level.get("different_wsi") or {}).get("mean"))
    if same is None or diff is None:
        return math.nan
    return float(same) - float(diff)


def finite_issue_notes(item: StepHealth) -> list[str]:
    notes = []
    for stage, values in item.signal_debug.items():
        finite = values.get("finite", math.nan)
        if math.isfinite(finite) and finite < 1.0:
            notes.append(f"{stage} finite={finite:.6f} < 1.0")
    return notes


def build_region_alignment(item: StepHealth) -> dict[str, Any]:
    normal_active = ((item.regions.get("normal") or {}).get("active_q"))
    real_label_active = ((item.regions.get("real_label") or {}).get("active_q"))
    rows = {}
    for variant in VARIANT_ORDER:
        region = item.regions.get(variant)
        if not region:
            continue
        compare_to = "normal" if variant in FEATURE_ONLY_VARIANTS else None
        if variant == "real":
            compare_to = "real_label"
        baseline = normal_active if compare_to == "normal" else real_label_active
        active_delta = (
            abs(region["active_q"] - baseline)
            if baseline is not None and math.isfinite(baseline)
            else math.nan
        )
        rows[variant] = {
            "active_q": region["active_q"],
            "missing_q": region["missing_q"],
            "fallback_q": region["fallback_q"],
            "null_q": region["null_q"],
            "allowed_pairs": region["allowed_pairs"],
            "compare_to": compare_to,
            "active_delta": active_delta,
            "active_aligned": (
                bool(math.isfinite(active_delta) and active_delta <= 0.05)
                if compare_to is not None
                else None
            ),
        }
    return rows


def build_ratio_summary(item: StepHealth) -> dict[str, Any]:
    rows = {}
    for variant in VARIANT_ORDER:
        ratio = item.ratios.get(variant)
        if not ratio:
            continue
        rows[variant] = {
            "min": ratio["ratio_min"],
            "mean": ratio["ratio_mean"],
            "max": ratio["ratio_max"],
        }
    return rows


def build_finite_summary(item: StepHealth) -> dict[str, Any]:
    finite_values = {
        stage: values.get("finite", math.nan)
        for stage, values in item.signal_debug.items()
    }
    finite_ok = all(
        not math.isfinite(value) or value >= 1.0
        for value in finite_values.values()
    ) and not item.numeric_anomalies
    return {
        "ok": finite_ok,
        "signal_finite": finite_values,
        "numeric_anomaly_count": len(item.numeric_anomalies),
    }


def gate_instrument(per_step: dict[str, Any]) -> dict[str, Any]:
    candidates = [per_step.get("500"), per_step.get("1000")]
    candidates = [item for item in candidates if item]
    if not candidates:
        return {"status": "WARN", "note": "no 500/1000 step health logs"}
    if any(item["status"] == "FAIL" for item in candidates):
        return {"status": "FAIL", "note": "instrument gate failed"}
    if any(item["status"] == "WARN" for item in candidates):
        return {"status": "WARN", "note": "instrument gate has warnings"}
    return {"status": "PASS", "note": "instrument gate passed"}


def gate_early_learning(per_step: dict[str, Any]) -> dict[str, Any]:
    candidates = [per_step.get("2000"), per_step.get("3000")]
    candidates = [item for item in candidates if item]
    if not candidates:
        return {"status": "WARN", "note": "no 2000/3000 step health logs"}
    has_feature_margin = any(
        any(math.isfinite(value) and value >= 0.05 for value in item.get("feature_margins", {}).values())
        for item in candidates
    )
    has_feature_gap = any(
        (item.get("reference", {}).get("real_feature", {}).get("loss_gap", 0.0) or 0.0) > 0.0
        for item in candidates
    )
    if not has_feature_margin and not has_feature_gap:
        return {
            "status": "WARN",
            "note": "no positive real_feature gap or feature margin observed yet",
        }
    return {"status": "PASS", "note": "early learning signal present"}


def gate_stop_loss(per_step: dict[str, Any]) -> dict[str, Any]:
    item = per_step.get("5000")
    if not item:
        return {"status": "WARN", "note": "no 5000 step health logs"}
    if item["status"] == "FAIL":
        return {"status": "FAIL", "note": "5000 step stop-loss failed"}
    return {"status": item["status"], "note": "use generated grid/top-1 retrieval for final 5k decision"}


def worst_status(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Cross V1 training health: {report['overall']}",
        f"latest_logged_step={report['latest_logged_step']}",
    ]
    lines.append("Gates:")
    for name, gate in report["gates"].items():
        lines.append(f"  {name}: {gate['status']} - {gate['note']}")
    lines.append("Steps:")
    for step, item in report["steps"].items():
        logged = item.get("step")
        lines.append(f"  target={step} logged={logged} status={item['status']}")
        finite = item.get("finite_summary") or {}
        if finite:
            anomaly_count = finite.get("numeric_anomaly_count", 0)
            ok_text = "ok" if finite.get("ok") else "BAD"
            signal_text = format_signal_finite(finite.get("signal_finite") or {})
            lines.append(
                f"    finite: {ok_text} anomalies={anomaly_count}"
                + (f" signal={signal_text}" if signal_text else "")
            )
        for anomaly in item.get("numeric_anomalies", []):
            lines.append(f"    numeric anomaly: {anomaly}")
        region_text = format_region_alignment(item.get("region_alignment") or {})
        if region_text:
            lines.append("    active_q alignment:")
            lines.extend(f"      {line}" for line in region_text)
        ratio_text = format_ratio_summary(item.get("ratio_summary") or {})
        if ratio_text:
            lines.append("    ratio min/mean/max:")
            lines.extend(f"      {line}" for line in ratio_text)
        if item.get("feature_margins"):
            margin_text = ", ".join(
                f"{stage}={value:.4f}" for stage, value in item["feature_margins"].items()
                if math.isfinite(value)
            )
            lines.append(f"    feature_margins: {margin_text}")
        for note in item.get("notes", [])[:6]:
            lines.append(f"    - {note}")
    return "\n".join(lines)


def format_signal_finite(values: dict[str, float]) -> str:
    parts = []
    for stage in sorted(values):
        value = values[stage]
        if math.isfinite(value):
            parts.append(f"{stage}={value:.3f}")
    return ", ".join(parts)


def format_region_alignment(rows: dict[str, Any]) -> list[str]:
    lines = []
    for variant in VARIANT_ORDER:
        row = rows.get(variant)
        if not row:
            continue
        compare_to = row.get("compare_to")
        delta = row.get("active_delta", math.nan)
        if compare_to:
            aligned = "OK" if row.get("active_aligned") else "WARN"
            suffix = f" Δactive_vs_{compare_to}={format_float(delta)} {aligned}"
        else:
            suffix = ""
        lines.append(
            f"{variant}: active={row['active_q']:.3f} missing={row['missing_q']:.3f} "
            f"fallback={row['fallback_q']:.3f} null={row['null_q']:.3f} "
            f"allowed={row['allowed_pairs']:.5f}{suffix}"
        )
    return lines


def format_ratio_summary(rows: dict[str, Any]) -> list[str]:
    lines = []
    for variant in VARIANT_ORDER:
        row = rows.get(variant)
        if not row:
            continue
        lines.append(
            f"{variant}: {row['min']:.3e}/{row['mean']:.3e}/{row['max']:.3e}"
        )
    return lines


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metrics(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload.get("final", payload))


def _small_component_fraction(metrics: dict[str, object], threshold: int = 16) -> float:
    fragmentation = metrics.get("fragmentation", {})
    overall = fragmentation.get("overall", {}) if isinstance(fragmentation, dict) else {}
    return float(overall.get(f"component_fraction_lt_{threshold}", float("nan")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare segmentator ablations against the agreed acceptance gates.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--runtime-cellvit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = _metrics(args.baseline)
    improved = _metrics(args.improved)
    boundary_gain = float(improved["boundary_f1_4"]) - float(baseline["boundary_f1_4"])
    hd95_reduction = (float(baseline["hd95"]) - float(improved["hd95"])) / max(float(baseline["hd95"]), 1e-8)
    baseline_small = _small_component_fraction(baseline)
    improved_small = _small_component_fraction(improved)
    fragmentation_reduction = (baseline_small - improved_small) / max(baseline_small, 1e-8)
    report: dict[str, object] = {
        "improved_vs_baseline": {
            "miou_delta": float(improved["mIoU"]) - float(baseline["mIoU"]),
            "boundary_f1_4_delta": boundary_gain,
            "hd95_reduction": hd95_reduction,
            "small_component_reduction": fragmentation_reduction,
            "passes": {
                "miou": float(improved["mIoU"]) >= float(baseline["mIoU"]) - 0.005,
                "boundary": boundary_gain >= 0.03,
                "hd95": hd95_reduction >= 0.10,
                "fragmentation": fragmentation_reduction >= 0.20,
            },
        }
    }
    if args.teacher and args.runtime_cellvit:
        teacher = _metrics(args.teacher)
        runtime = _metrics(args.runtime_cellvit)
        miou_gain = float(runtime["mIoU"]) - float(teacher["mIoU"])
        runtime_boundary_gain = float(runtime["boundary_f1_4"]) - float(teacher["boundary_f1_4"])
        report["runtime_cellvit_gate"] = {
            "miou_delta": miou_gain,
            "boundary_f1_4_delta": runtime_boundary_gain,
            "deploy_runtime_cellvit": miou_gain >= 0.015 or runtime_boundary_gain >= 0.03,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

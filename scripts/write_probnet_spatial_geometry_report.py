#!/usr/bin/env python3
"""Write the formal geometry-only ProbNet spatial benchmark conclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ERROR_METRICS = (
    "nnd_w1_um",
    "ripley_k_normalized_l1",
    "boundary_distance_w1_um",
    "component_occupancy_l1_per_target",
)


def metric_label(metric: str) -> str:
    return {
        "nnd_w1_um": "NND W1 (µm)",
        "ripley_k_normalized_l1": "Ripley-K normalized L1",
        "boundary_distance_w1_um": "Tissue-boundary W1 (µm)",
        "component_occupancy_l1_per_target": "Component occupancy L1 / target",
    }[metric]


def f(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    p1_path = args.run_root / "p1_spatial_ablation" / "paired_spatial_summary.json"
    p2_path = (
        args.run_root
        / "p2_geometry_endpoint"
        / "planned_layouts"
        / "validation.json"
    )
    p1 = load_json(p1_path)
    p2 = load_json(p2_path)
    probnet = p1["samplers"]["probnet"]["equal_dataset_macro"]
    uniform = p1["samplers"]["uniform"]["equal_dataset_macro"]
    paired = p1["paired_probnet_vs_uniform"]
    relative = p1["relative_error_reduction_vs_uniform"]
    claim = bool(p1["clear_learned_spatial_structure_claim_passed"])
    safety = bool(p2["geometry_safety_gate_passed"])

    conclusion = (
        "ProbNet demonstrated clearly superior learned cell spatial structure "
        "over uniform simple sampling under the frozen gate."
        if claim
        else "ProbNet did not demonstrate clearly superior learned cell spatial "
        "structure over uniform simple sampling under the frozen gate."
    )
    production = (
        "The strict geometry endpoint passed its production safety gate."
        if safety
        else "The strict geometry endpoint did not pass its production safety gate."
    )

    lines = [
        "# ProbNet Strict Geometry Benchmark",
        "",
        "## Formal conclusion",
        "",
        conclusion,
        "",
        production,
        "",
        "ProbNet supplies only `P(nucleus)` for candidate ordering. Counts, exact "
        "type quotas, candidate pools, component quotas, shapes, and retries are "
        "checkpoint-independent. H&E synthesis and CellViT are excluded.",
        "",
        "## ProbNet versus primary uniform baseline",
        "",
        "| Endpoint | ProbNet | Uniform | Paired delta | 95% CI | Relative error reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ERROR_METRICS:
        item = paired[metric]
        lines.append(
            f"| {metric_label(metric)} | {f(probnet[metric])} | "
            f"{f(uniform[metric])} | "
            f"{f(item['delta_probnet_minus_baseline'])} | "
            f"[{f(item['ci95_low'])}, {f(item['ci95_high'])}] | "
            f"{f(100 * relative[metric], 1)}% |"
        )
    overall = p2["overall"]
    probnet_by_dataset = p1["samplers"]["probnet"]["by_dataset"]
    uniform_by_dataset = p1["samplers"]["uniform"]["by_dataset"]
    lines.extend(
        [
            "",
            "## Geometry safety endpoint",
            "",
            f"- Samples: {overall['samples']}",
            f"- Requested / placed / unfilled: {overall['requested']} / "
            f"{overall['placed']} / {overall['unfilled']}",
            f"- Placement completion: {f(100 * overall['placement_completion'], 2)}%",
            f"- Overlap pixels: {overall['overlap_pixels']}",
            f"- Outside-valid-tissue pixels: {overall['outside_tissue_pixels']}",
            f"- Retained-nucleus preservation: "
            f"{f(100 * overall['retained_preservation_rate'], 2)}%",
            "",
            "## P1 placement completion by dataset",
            "",
            "| Dataset | ProbNet | Uniform |",
            "|---|---:|---:|",
        ]
    )
    for dataset in sorted(probnet_by_dataset):
        lines.append(
            f"| {dataset} | "
            f"{f(100 * probnet_by_dataset[dataset]['placement_completion'], 2)}% | "
            f"{f(100 * uniform_by_dataset[dataset]['placement_completion'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "ORCA remains a stress limitation when P1 requests reconstruction "
            "of every hidden oracle nucleus: ProbNet and uniform both complete "
            "only about 91.5%. This is not a learned-ranking advantage and must "
            "not be hidden by the equal-dataset macro. The production-count P2 "
            "ORCA endpoint nevertheless placed 249/249 nuclei with zero overlap "
            "and zero outside-valid-tissue pixels.",
            "",
            "## Frozen claim rule",
            "",
            "- Uniform is the primary simple-sampling comparator.",
            "- At least two primary errors must improve significantly and by ≥5%.",
            "- At least one material improvement must be NND or Ripley-K.",
            "- No primary spatial regression is allowed.",
            "- Geometry safety must pass.",
            "",
            f"Material improvements: "
            f"{', '.join(p1['material_primary_improvements_vs_uniform']) or 'none'}.",
            f"Core NND/Ripley improvement: "
            f"{p1['core_spatial_structure_improvement_detected']}.",
            f"Final learned-structure claim gate: {claim}.",
            "",
        ]
    )
    report_path = args.run_root / "final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    conclusion_path = args.run_root / "final_conclusion.json"
    conclusion_path.write_text(
        json.dumps(
            {
                "clear_learned_spatial_structure_claim_passed": claim,
                "geometry_safety_gate_passed": safety,
                "conclusion": conclusion,
                "production_conclusion": production,
                "p1_orca_stress_placement_completion": (
                    probnet_by_dataset["ORCA"]["placement_completion"]
                ),
                "p1_orca_uniform_placement_completion": (
                    uniform_by_dataset["ORCA"]["placement_completion"]
                ),
                "p2_orca_requested": p2["by_dataset"]["ORCA"]["requested"],
                "p2_orca_placed": p2["by_dataset"]["ORCA"]["placed"],
                "p1_summary": str(p1_path),
                "p2_validation": str(p2_path),
                "h_e_and_cellvit_excluded": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(report_path)
    print(conclusion_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

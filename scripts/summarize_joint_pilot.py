#!/usr/bin/env python3
"""Summarize an auditable joint-condition shadow run by failure boundary."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contexts = {item["case_id"]: item for item in manifest}
    run = json.loads((args.run_root / "joint_run_summary.json").read_text(encoding="utf-8"))
    rows = []
    status_counts = Counter()
    reason_counts = Counter()
    domain_counts = defaultdict(Counter)
    total_tokens = 0
    api_providers = set()
    funnel_counts = Counter()
    for item in run:
        case_id = item["case_id"]
        context = contexts[case_id]
        case_dir = Path(item["artifact_paths"]["case_dir"])
        result = _read(case_dir / "result.json", {})
        candidates = _read(case_dir / "candidates.json", [])
        gates = _read(case_dir / "joint_gate_reports.json", [])
        passing = sum(bool(report.get("passed")) for report in gates)
        tissue_reports = _read(case_dir / "tissue_gate_reports.json", [])
        if not tissue_reports:
            for report_path in sorted(case_dir.glob("tissue_gate_reports_pass_*.json")):
                tissue_reports.extend(_read(report_path, []))
        nuclei_preflight = _read(case_dir / "joint_nuclei_preflight.json", {})
        tissue_execution = []
        for execution_path in sorted(
            case_dir.glob("tissue_execution_contract_pass_*.json")
        ):
            tissue_execution.append(_read(execution_path, {}))
        reasons = item.get("abstain_reasons", [])
        reason_class = _classify(reasons[0] if reasons else "")
        # Current runs persist every tissue gate invocation. Derive the funnel
        # from artifacts rather than inferring that an upstream data failure
        # must have occurred after tissue execution.
        tissue_gate_ran = bool(tissue_reports)
        tissue_gate_passed = any(
            bool(report.get("passed")) for report in tissue_reports
        )
        stages = {
            "input_scene_built": (
                (case_dir / "joint_scene_graph.json").is_file()
                or (case_dir / "source_joint_overlay.png").is_file()
            ),
            "nuclei_preflight_run": bool(nuclei_preflight),
            "nuclei_preflight_passed": bool(
                nuclei_preflight.get("feasible_interface_ids")
                and not nuclei_preflight.get("required_auxiliary_missing")
                and not nuclei_preflight.get("required_provenance_missing")
            ),
            "tissue_gate_run": tissue_gate_ran,
            "tissue_gate_passed": tissue_gate_passed,
            "tissue_execution_certified": any(
                value.get("certified_candidate_ids") for value in tissue_execution
            ),
            "cell_candidates_generated": bool(candidates),
            "joint_gate_passed": passing > 0,
            "critic_run": (case_dir / "joint_critic.json").is_file(),
            "selected": item.get("selected_candidate_id") is not None,
        }
        funnel_counts.update(key for key, value in stages.items() if value)
        failed_gates = Counter(
            check["check_id"]
            for report in gates
            for check in report.get("checks", [])
            if not check.get("passed")
        )
        status_counts[item["status"]] += 1
        reason_counts[reason_class] += 1
        domain = context["pathology_domain_id"]
        domain_counts[domain][reason_class] += 1
        usage = result.get("usage", {})
        for value in usage.values():
            for current in _walk_dicts(value):
                total_tokens += int(current.get("input_tokens", 0) or 0)
                total_tokens += int(current.get("output_tokens", 0) or 0)
                provider = current.get("provider")
                if isinstance(provider, str) and provider.startswith("openai"):
                    api_providers.add(provider)
        rows.append(
            {
                "case_id": case_id,
                "pathology_domain_id": domain,
                "annotation_profile_id": context["annotation_profile_id"],
                "primitive_id": context["primitive_id"],
                "mechanism_id": context["provenance"].get("joint_mechanism_id"),
                "status": item["status"],
                "reason_class": reason_class,
                "reasons": reasons,
                "candidate_count": len(candidates),
                "gate_passing_candidate_count": passing,
                "failed_gate_counts": dict(sorted(failed_gates.items())),
                "review_board": item["artifact_paths"].get("joint_condition_review"),
                "stages": stages,
            }
        )
    payload = {
        "schema_version": "joint-pilot-summary-v1",
        "cases": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "funnel_counts": {
            key: funnel_counts.get(key, 0)
            for key in (
                "input_scene_built",
                "nuclei_preflight_run",
                "nuclei_preflight_passed",
                "tissue_gate_run",
                "tissue_gate_passed",
                "tissue_execution_certified",
                "cell_candidates_generated",
                "joint_gate_passed",
                "critic_run",
                "selected",
            )
        },
        "domain_reason_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(domain_counts.items())
        },
        "network_audit": {
            "api_provider_ids": sorted(api_providers),
            "reported_total_tokens": total_tokens,
            "no_api_usage_reported": not api_providers and total_tokens == 0,
        },
        "rows": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_output.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "json": str(args.json_output), "markdown": str(args.markdown_output)}))
    return 0


def _classify(reason: str) -> str:
    if "offline visual Planner abstained" in reason:
        return "manual_visual_planner_abstain"
    if "hard minimum" in reason:
        return "safe_tissue_capacity_below_14pct"
    if "no topology-safe edit pixels" in reason:
        return "no_topology_safe_tissue_pixels"
    if "required auxiliary" in reason or "auxiliary observations" in reason:
        return "missing_auxiliary_map"
    if "required annotation provenance" in reason:
        return "missing_annotation_provenance"
    if "preflight found no interface" in reason:
        return "no_joint_preflight_capacity"
    if "native nucleus instances" in reason:
        return "missing_native_nucleus_instances"
    if "no tissue candidate" in reason:
        return "no_tissue_candidate_passed"
    if "no paired tissue--cell" in reason:
        return "joint_gates_failed"
    return reason or "none"


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _markdown(payload: dict) -> str:
    lines = [
        "# G2-50 Joint Condition Shadow Summary",
        "",
        f"Cases: {payload['cases']}. Statuses: {payload['status_counts']}.",
        f"No API usage reported: {payload['network_audit']['no_api_usage_reported']}.",
        "",
        "## Failure boundaries",
        "",
    ]
    lines.extend(
        f"- `{key}`: {value}"
        for key, value in payload["reason_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Stage funnel",
            "",
            *(
                f"- `{key}`: {value}/{payload['cases']}"
                for key, value in payload["funnel_counts"].items()
            ),
            "",
            "## Cases",
            "",
            "| Case | Domain | Primitive | Mechanism | Status boundary | Candidates/pass | Failed joint gates |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for row in payload["rows"]:
        failed = ", ".join(
            f"{key}:{value}" for key, value in row["failed_gate_counts"].items()
        ) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["pathology_domain_id"],
                    row["primitive_id"],
                    str(row["mechanism_id"]),
                    row["reason_class"],
                    f"{row['candidate_count']}/{row['gate_passing_candidate_count']}",
                    failed,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

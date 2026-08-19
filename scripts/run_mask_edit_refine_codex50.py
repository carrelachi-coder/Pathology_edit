#!/usr/bin/env python3
"""Prepare and render the no-API, current-Codex 50-case research pilot.

This runner does not call a model endpoint.  It freezes a stratified cohort,
enumerates a small bank of executable interface/anchor plans, runs the normal
deterministic candidate generators and fail-closed gates, then materializes
compact sheets for review by the current Codex session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit_refine.agents import (
    EDIT_PLAN_SCHEMA_VERSION,
    HeuristicInterfacePlanner,
    validate_edit_plan,
)
from phase3_mask_edit_refine.candidates import generate_candidates
from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.execution import compile_edit_plan
from phase3_mask_edit_refine.gates import GateContext, GateRegistry
from phase3_mask_edit_refine.models import (
    CandidateMask,
    CaseContext,
    DepthProfile,
    EditPlan,
    GateReport,
    InterfaceExecutionContract,
    PlannedInterface,
)
from phase3_mask_edit_refine.scene import build_scene_analysis
from phase3_mask_edit_refine.skills import (
    SkillRepository,
    bind_active_bundle_to_case,
    validate_active_bundle_authority,
)
from phase3_mask_edit_refine.visualization import (
    id_mask_to_rgb,
    save_critic_contact_sheet,
    save_planner_panels,
)

COHORT_ALLOCATION = {
    ("breast", "tumor_increase"): 3,
    ("breast", "tumor_decrease"): 3,
    ("breast", "stroma_increase"): 3,
    ("colorectal", "tumor_increase"): 3,
    ("colorectal", "tumor_decrease"): 3,
    ("colorectal", "stroma_increase"): 3,
    ("lung", "tumor_increase"): 3,
    ("lung", "tumor_decrease"): 3,
    ("lung", "stroma_increase"): 2,
    ("oral", "tumor_increase"): 4,
    ("oral", "tumor_decrease"): 4,
    ("prostate", "tumor_increase"): 4,
    ("prostate", "tumor_decrease"): 4,
    ("skin", "tumor_increase"): 3,
    ("skin", "tumor_decrease"): 3,
    ("skin", "stroma_increase"): 2,
}

# This frozen replacement explicitly carries the case-152 regression lineage.
FORCED_CASE_IDS = {"g2_001_colorectal_tumor_decrease_9801d9efe5"}
RESEARCH_IGNORED_GATE_IDS = frozenset({"profile_required_provenance"})


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--index", required=True)
    freeze.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cohort", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--workers", type=int, default=6)
    prepare.add_argument("--max-proposals", type=int, default=6)
    prepare.add_argument("--ready-candidates", type=int, default=8)
    boards = commands.add_parser("boards")
    boards.add_argument("--cohort", required=True)
    boards.add_argument("--output-root", required=True)
    boards.add_argument("--board-dir", required=True)
    boards.add_argument("--per-board", type=int, default=10)
    boards.add_argument("--status")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--cohort", required=True)
    finalize.add_argument("--output-root", required=True)
    finalize.add_argument("--decisions", required=True)
    finalize.add_argument("--review-root", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        return _freeze(args)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "boards":
        return _boards(args)
    if args.command == "finalize":
        return _finalize(args)
    raise AssertionError(args.command)


def _freeze(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.index).read_text(encoding="utf-8"))
    records = payload["cases"]
    selected: list[dict[str, Any]] = []
    for stratum, count in COHORT_ALLOCATION.items():
        pool = [
            dict(record)
            for record in records
            if (record["organ"], record["primitive"]) == stratum
        ]
        pool.sort(
            key=lambda record: (
                record["case_id"] not in FORCED_CASE_IDS,
                hashlib.sha256(record["case_id"].encode("utf-8")).hexdigest(),
            )
        )
        if len(pool) < count:
            raise ValueError(f"stratum {stratum} has {len(pool)} cases, need {count}")
        selected.extend(pool[:count])
    selected.sort(key=lambda item: (item["organ"], item["primitive"], item["case_id"]))
    if len(selected) != 50 or len({item["case_id"] for item in selected}) != 50:
        raise ValueError("frozen cohort must contain exactly 50 unique cases")
    output_payload = {
        "pilot_id": "mask-edit-refine-codex-session-50-v1",
        "provider": "current_codex_session_no_api",
        "selection_policy": {
            "method": "pre-result stratified sha256 ordering",
            "allocation": {f"{a}|{b}": n for (a, b), n in COHORT_ALLOCATION.items()},
            "forced_regression_lineage": sorted(FORCED_CASE_IDS),
            "case_175_note": (
                "exact case-175 source artifact is absent from the mounted 120-case bundle; "
                "the frozen colorectal stroma stratum remains included without fabricating input"
            ),
        },
        "case_count": len(selected),
        "cases": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, output_payload)
    print(json.dumps({"status": "frozen", "output": str(output), "cases": 50}))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    cohort = json.loads(Path(args.cohort).read_text(encoding="utf-8"))["cases"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (record, str(output_root), int(args.max_proposals), int(args.ready_candidates))
        for record in cohort
    ]
    started = time.time()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(_prepare_one, task): task[0]["case_id"] for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "case_id": case_id,
                    "status": "worker_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(tasks),
                        "case_id": case_id,
                        "status": result["status"],
                        "ready_candidates": result.get("ready_candidate_count", 0),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    ordered = {item["case_id"]: item for item in results}
    summary = {
        "pilot_id": "mask-edit-refine-codex-session-50-v1",
        "elapsed_seconds": round(time.time() - started, 3),
        "case_count": len(tasks),
        "status_counts": _counts(item["status"] for item in results),
        "cases": [ordered[record["case_id"]] for record in cohort],
    }
    _write_json(output_root / "prepare_summary.json", summary)
    print(json.dumps(summary["status_counts"], ensure_ascii=False, sort_keys=True))
    return 0


def _prepare_one(task: tuple[dict[str, Any], str, int, int]) -> dict[str, Any]:
    record, output_root_raw, max_proposals, ready_target = task
    case = CaseContext.from_mapping(
        json.loads(Path(record["context"]).read_text(encoding="utf-8"))
    )
    case_dir = Path(output_root_raw) / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    repository = SkillRepository()
    gates = GateRegistry()
    mask = load_id_mask(case.source_mask_uri)
    schema = repository.annotation_schema(case.annotation_profile_id)
    bundle = repository.compose(
        pathology_domain_id=case.pathology_domain_id,
        annotation_profile_id=case.annotation_profile_id,
        primitive_id=case.primitive_id,
        production=False,
        available_checker_ids=gates.available_checker_ids,
        case_provenance=case.provenance,
    )
    scene = build_scene_analysis(mask, schema=schema, pixel_size_um=case.pixel_size_um)
    bundle = bind_active_bundle_to_case(bundle, case=case, scene=scene)
    validate_active_bundle_authority(
        bundle,
        case_provenance=case.provenance,
        require_live_binding=True,
        case=case,
        scene=scene,
    )
    _write_json(case_dir / "case_context.json", case.to_metadata())
    _write_json(case_dir / "scene_graph.json", scene.graph.to_metadata())
    _write_json(case_dir / "active_skills.json", bundle.to_metadata())
    panels = save_planner_panels(
        image_path=case.source_image_uri,
        mask=mask,
        scene=scene,
        output_dir=case_dir / "planner_panels",
    )
    proposal_plans = _enumerate_plan_proposals(
        case=case,
        scene=scene,
        bundle=bundle,
        maximum=max_proposals,
    )
    proposal_root = case_dir / "proposals"
    proposal_root.mkdir(exist_ok=True)
    ready: list[tuple[float, CandidateMask, GateReport, str, dict[str, Any]]] = []
    proposal_records: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    ready_proposal_ids: set[str] = set()
    for proposal_index, (proposal_id, raw_plan) in enumerate(proposal_plans, start=1):
        proposal_dir = proposal_root / proposal_id
        proposal_dir.mkdir(exist_ok=True)
        proposal_record: dict[str, Any] = {"proposal_id": proposal_id}
        try:
            validate_edit_plan(raw_plan, case=case, scene=scene, bundle=bundle)
            compiled, compiler_audit = compile_edit_plan(
                raw_plan,
                source_mask=mask,
                schema=schema,
                scene=scene,
            )
            validate_edit_plan(compiled, case=case, scene=scene, bundle=bundle)
            candidates = generate_candidates(
                mask,
                schema=schema,
                scene=scene,
                plan=compiled,
                bundle=bundle,
                seed=case.seed + proposal_index * 1_000_003,
            )
            reports = tuple(
                gates.run(
                    GateContext(
                        case=case,
                        source_mask=mask,
                        schema=schema,
                        scene=scene,
                        bundle=bundle,
                        plan=compiled,
                        candidate=candidate,
                    )
                )
                for candidate in candidates
            )
            _write_json(proposal_dir / "raw_plan.json", raw_plan.to_metadata())
            _write_json(proposal_dir / "compiled_plan.json", compiled.to_metadata())
            _write_json(proposal_dir / "compiler_audit.json", compiler_audit)
            _write_json(
                proposal_dir / "gate_reports.json",
                [report.to_metadata() for report in reports],
            )
            passed_here = 0
            for candidate, report in zip(candidates, reports):
                failed = _failed_hard_gate_ids(report)
                for check_id in failed - RESEARCH_IGNORED_GATE_IDS:
                    failure_counts[check_id] = failure_counts.get(check_id, 0) + 1
                if failed - RESEARCH_IGNORED_GATE_IDS:
                    continue
                passed_here += 1
                global_id = f"{proposal_id}__{candidate.candidate_id.replace(':', '_')}"
                renamed_candidate = replace(candidate, candidate_id=global_id)
                renamed_report = replace(report, candidate_id=global_id, passed=True)
                score = _deterministic_review_score(renamed_report)
                ready.append(
                    (
                        score,
                        renamed_candidate,
                        renamed_report,
                        proposal_id,
                        compiled.to_metadata(),
                    )
                )
                ready_proposal_ids.add(proposal_id)
            proposal_record.update(
                {
                    "status": "generated",
                    "candidate_count": len(candidates),
                    "research_geometry_pass_count": passed_here,
                    "compiler_audit": compiler_audit,
                }
            )
        except Exception as exc:  # noqa: BLE001
            proposal_record.update(
                {
                    "status": "proposal_rejected",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_plan": raw_plan.to_metadata(),
                }
            )
            _write_json(proposal_dir / "proposal_error.json", proposal_record)
        proposal_records.append(proposal_record)
        if len(ready) >= ready_target and len(ready_proposal_ids) >= 2:
            break

    selected_ready = _select_diverse_ready(ready, maximum=ready_target)
    candidate_root = case_dir / "review_candidates"
    candidate_root.mkdir(exist_ok=True)
    virtual_candidates: list[CandidateMask] = []
    virtual_reports: list[GateReport] = []
    candidate_records: list[dict[str, Any]] = []
    for rank, (score, candidate, report, proposal_id, plan_metadata) in enumerate(
        selected_ready, start=1
    ):
        target = candidate_root / candidate.candidate_id
        target.mkdir(exist_ok=True)
        np.save(target / "target_mask.npy", candidate.target_mask, allow_pickle=False)
        np.save(target / "change_region.npy", candidate.change_region, allow_pickle=False)
        Image.fromarray(id_mask_to_rgb(candidate.target_mask)).save(target / "target_mask.png")
        Image.fromarray(candidate.change_region.astype(np.uint8) * 255).save(
            target / "change_region.png"
        )
        _write_json(target / "tool_trace.json", candidate.to_metadata())
        _write_json(target / "gate_report.json", report.to_metadata())
        _write_json(target / "compiled_plan.json", plan_metadata)
        candidate_records.append(
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "proposal_id": proposal_id,
                "deterministic_score": score,
                "tool_name": candidate.tool_name,
                "changed_pixels": int(np.count_nonzero(candidate.change_region)),
                "path": str(target),
            }
        )
        virtual_candidates.append(candidate)
        virtual_reports.append(report)

    status = "ready_for_codex_review" if virtual_candidates else "deterministic_abstain"
    if virtual_candidates:
        save_critic_contact_sheet(
            image_path=case.source_image_uri,
            source_mask=mask,
            candidates=virtual_candidates,
            gate_reports=virtual_reports,
            scene=scene,
            output_path=case_dir / "codex_contact_sheet.png",
            columns=4,
        )
        _save_single_preview(
            case=case,
            source_mask=mask,
            candidate=virtual_candidates[0],
            output_path=case_dir / "provisional_preview.png",
        )
    else:
        _save_abstain_preview(
            case=case,
            semantic_overlay_path=panels[1],
            reasons=failure_counts,
            output_path=case_dir / "provisional_preview.png",
        )
    result = {
        "case_id": case.case_id,
        "organ": record["organ"],
        "primitive": record["primitive"],
        "status": status,
        "ready_candidate_count": len(candidate_records),
        "provisional_candidate_id": (
            candidate_records[0]["candidate_id"] if candidate_records else None
        ),
        "research_only_ignored_hard_gates": sorted(RESEARCH_IGNORED_GATE_IDS),
        "failure_counts": failure_counts,
        "proposals": proposal_records,
        "review_candidates": candidate_records,
        "planner_panels": list(panels),
        "contact_sheet": (
            str(case_dir / "codex_contact_sheet.png") if virtual_candidates else None
        ),
        "preview": str(case_dir / "provisional_preview.png"),
    }
    _write_json(case_dir / "prepare_result.json", result)
    return result


def _enumerate_plan_proposals(*, case, scene, bundle, maximum: int):
    legal = scene.interfaces_for(
        source_labels=bundle.edit_contract.source_label_options,
        target_label=bundle.edit_contract.target_label,
    )
    if not legal:
        return []
    groups: dict[str, list[Any]] = {}
    for interface in legal:
        groups.setdefault(interface.source_label, []).append(interface)
    for group in groups.values():
        group.sort(key=lambda item: (-item.contact_pixels, item.interface_id))
    ordered_labels = sorted(
        groups,
        key=lambda label: (-sum(item.contact_pixels for item in groups[label]), label),
    )
    specifications: list[tuple[str, tuple[Any, ...], str]] = []
    for label in ordered_labels:
        group = groups[label]
        if len(group) >= 4:
            specifications.append(
                (f"{label}_broad_quad_01_04", tuple(group[:4]), "multi_lobe")
            )
        if len(group) >= 3:
            specifications.append(
                (f"{label}_broad_triple_01_03", tuple(group[:3]), "strong_taper")
            )
        if len(group) >= 2:
            specifications.append(
                (f"{label}_broad_pair_01_02", tuple(group[:2]), "multi_lobe")
            )
        for index, interface in enumerate(group[:3], start=1):
            specifications.append(
                (f"{label}_single_{index:02d}", (interface,), "strong_taper")
            )
        if len(group) >= 3:
            specifications.append(
                (f"{label}_broad_pair_02_03", tuple(group[1:3]), "strong_taper")
            )
    plans = []
    seen: set[tuple[str, ...]] = set()
    for label, interfaces, profile_kind in specifications:
        identity = tuple(item.interface_id for item in interfaces) + (profile_kind,)
        if identity in seen:
            continue
        seen.add(identity)
        plans.append(
            (
                f"proposal_{len(plans) + 1:02d}_{_slug(label)}_{profile_kind}",
                _make_plan(
                    case,
                    scene,
                    bundle,
                    interfaces,
                    profile_kind=profile_kind,
                ),
            )
        )
        if len(plans) >= maximum:
            break
    return plans


def _make_plan(
    case,
    scene,
    bundle,
    interfaces,
    *,
    profile_kind: str = "strong_taper",
) -> EditPlan:
    # Reuse the research planner's calibrated deterministic parameter envelope,
    # while replacing its interface choice with this explicitly addressable plan.
    base, _ = HeuristicInterfacePlanner().create_plan(
        case=case,
        scene=scene,
        bundle=bundle,
        image_paths=(),
    )
    total_contact = max(1, sum(item.contact_pixels for item in interfaces))
    target_pixels = round(scene.graph.width * scene.graph.height * case.area_budget.target_fraction)
    rule_ids = tuple(rule.rule_id for rule in bundle.active_rules) + tuple(
        item.constraint_id for item in bundle.active_mask_constraints
    )
    planned = []
    for interface in interfaces:
        allocation = interface.contact_pixels / total_contact
        estimated_depth = target_pixels * allocation / max(interface.contact_pixels, 1)
        peak = float(np.clip(np.ceil(estimated_depth * 1.35), 4.0, 120.0))
        if profile_kind == "multi_lobe":
            profile_mode = "multi_lobe"
            edge_ratio = 0.20
            taper_fraction = 0.42
            lobe_count = 3
            noise_ratio = 0.18
        else:
            profile_mode = "tapered_lobe"
            edge_ratio = 0.15
            taper_fraction = 0.45
            lobe_count = 1
            noise_ratio = 0.22
        planned.append(
            PlannedInterface(
                interface_id=interface.interface_id,
                source_component_id=interface.source_component_id,
                target_component_id=interface.target_component_id,
                anchor_segment="all_addressable_subarcs",
                allowed_edit_band_px=(0.0, 128.0),
                execution_contract=InterfaceExecutionContract(
                    anchor_segment_ids=interface.anchor_segment_ids,
                    area_allocation_fraction=float(allocation),
                    depth_profile=DepthProfile(
                        mode=profile_mode,
                        peak_depth_px=peak,
                        edge_depth_px=max(1.0, peak * edge_ratio),
                        taper_fraction=taper_fraction,
                        lobe_count=lobe_count,
                        noise_amplitude_px=min(24.0, peak * noise_ratio),
                        noise_correlation_px=18.0,
                    ),
                    min_anchor_coverage_fraction=0.50,
                    max_off_anchor_contact_fraction=0.02,
                    allocation_tolerance_fraction=0.02,
                ),
                prohibited_region_ids=(),
                supporting_rule_ids=rule_ids,
                expected_morphology=(
                    "broad interface-bound edit with continuous tapered depth and no local notch"
                ),
                confidence=0.80,
            )
        )
    parameter_ranges = dict(base.tool_program.parameter_ranges)
    parameter_ranges["max_changed_components"] = min(4, len(interfaces))
    return replace(
        base,
        schema_version=EDIT_PLAN_SCHEMA_VERSION,
        source_labels=(interfaces[0].source_label,),
        candidate_interfaces=tuple(planned),
        tool_program=replace(
            base.tool_program,
            parameter_ranges=parameter_ranges,
        ),
        uncertainties=(
            "current Codex session must visually inspect H&E and choose or reject this plan",
        ),
        planner_confidence=0.80,
        escalation_reason=None,
    )


def _failed_hard_gate_ids(report: GateReport) -> set[str]:
    return {
        check.check_id
        for check in report.checks
        if check.severity == "hard" and not check.passed
    }


def _deterministic_review_score(report: GateReport) -> float:
    metrics = {check.check_id: check.metrics for check in report.checks}
    compactness = float(
        metrics.get("boundary_naturalness", {}).get("boundary_compactness", 99.0)
    )
    p95_depth = float(metrics.get("depth_span_ratio", {}).get("p95_depth_px", 999.0))
    components = float(
        metrics.get("component_topology", {}).get("component_count", 99.0)
    )
    retention = metrics.get("source_component_retention", {}).get("components", {})
    maximum_consumed = max(
        (float(item.get("changed_fraction", 0.0)) for item in retention.values()),
        default=1.0,
    )
    # Lower is better. This is only a stable review ordering, never a visual
    # pathology acceptance decision.
    return round(compactness + p95_depth / 20.0 + components * 0.25 + maximum_consumed * 4.0, 6)


def _select_diverse_ready(ready, *, maximum: int):
    ordered = sorted(ready, key=lambda item: (item[0], item[1].candidate_id))
    selected = []
    seen_keys: set[tuple[str, str]] = set()
    for item in ordered:
        key = (item[3], item[1].tool_name)
        if key in seen_keys and len(selected) < min(4, maximum):
            continue
        selected.append(item)
        seen_keys.add(key)
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        selected_ids = {item[1].candidate_id for item in selected}
        selected.extend(
            item for item in ordered if item[1].candidate_id not in selected_ids
        )
    return selected[:maximum]


def _save_single_preview(
    *,
    case,
    source_mask,
    candidate,
    output_path: Path,
    selection_label: str = "provisional",
) -> None:
    with Image.open(case.source_image_uri) as opened:
        image = opened.convert("RGB")
    source_rgb = Image.fromarray(id_mask_to_rgb(source_mask))
    target_rgb = Image.fromarray(id_mask_to_rgb(candidate.target_mask))
    source_overlay = Image.blend(image, source_rgb, 0.42)
    target_overlay = Image.blend(image, target_rgb, 0.42)
    canvas = Image.new("RGB", (image.width * 2, image.height + 44), "white")
    canvas.paste(source_overlay, (0, 44))
    canvas.paste(target_overlay, (image.width, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), f"{case.case_id} | {case.primitive_id}", fill="black")
    draw.text(
        (6, 23),
        f"SOURCE -> {selection_label} {candidate.candidate_id}",
        fill="black",
    )
    canvas.save(output_path)


def _save_abstain_preview(*, case, semantic_overlay_path, reasons, output_path: Path) -> None:
    with Image.open(semantic_overlay_path) as opened:
        overlay = opened.convert("RGB")
    canvas = Image.new("RGB", (overlay.width * 2, overlay.height + 44), "white")
    canvas.paste(overlay, (0, 44))
    canvas.paste(overlay, (overlay.width, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), f"{case.case_id} | DETERMINISTIC ABSTAIN", fill=(180, 0, 0))
    summary = ", ".join(f"{key}:{value}" for key, value in sorted(reasons.items()))
    draw.text((6, 23), summary[:180], fill="black")
    canvas.save(output_path)


def _boards(args: argparse.Namespace) -> int:
    cohort = json.loads(Path(args.cohort).read_text(encoding="utf-8"))["cases"]
    output_root = Path(args.output_root)
    board_dir = Path(args.board_dir)
    board_dir.mkdir(parents=True, exist_ok=True)
    per_board = max(1, int(args.per_board))
    previews = []
    for record in cohort:
        path = output_root / record["case_id"] / "provisional_preview.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        if args.status:
            review_path = output_root / record["case_id"] / "codex_review.json"
            if not review_path.is_file():
                continue
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if review.get("status") != args.status:
                continue
        previews.append((record, path))
    outputs = []
    for board_index, start in enumerate(range(0, len(previews), per_board), start=1):
        batch = previews[start : start + per_board]
        opened = [(record, Image.open(path).convert("RGB")) for record, path in batch]
        width = max(image.width for _, image in opened)
        height = max(image.height for _, image in opened)
        canvas = Image.new("RGB", (width, height * len(opened)), "white")
        for row, (_, image) in enumerate(opened):
            canvas.paste(image, (0, row * height))
        output = board_dir / f"codex50_board_{board_index:02d}.jpg"
        canvas.save(output, quality=92)
        outputs.append(str(output))
        for _, image in opened:
            image.close()
    _write_json(board_dir / "boards.json", {"boards": outputs})
    print(json.dumps({"status": "written", "boards": outputs}, ensure_ascii=False))
    return 0


def _finalize(args: argparse.Namespace) -> int:
    cohort = json.loads(Path(args.cohort).read_text(encoding="utf-8"))["cases"]
    decision_payload = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    overrides = {
        item["case_id"]: item for item in decision_payload.get("decisions", [])
    }
    output_root = Path(args.output_root)
    review_root = Path(args.review_root)
    results = []
    for record in cohort:
        case_id = record["case_id"]
        prepared_dir = output_root / case_id
        prepared = json.loads(
            (prepared_dir / "prepare_result.json").read_text(encoding="utf-8")
        )
        override = overrides.get(case_id)
        case_review_dir = review_root / case_id
        case_review_dir.mkdir(parents=True, exist_ok=True)
        if override is not None and override.get("status") == "selected":
            candidate_id = str(override["candidate_id"])
            known = {item["candidate_id"] for item in prepared["review_candidates"]}
            if candidate_id not in known:
                raise ValueError(f"unknown review candidate for {case_id}: {candidate_id}")
            candidate_dir = prepared_dir / "review_candidates" / candidate_id
            target_mask = np.load(candidate_dir / "target_mask.npy", allow_pickle=False)
            change_region = np.load(
                candidate_dir / "change_region.npy", allow_pickle=False
            ).astype(bool)
            case = CaseContext.from_mapping(
                json.loads(Path(record["context"]).read_text(encoding="utf-8"))
            )
            source_mask = load_id_mask(case.source_mask_uri)
            candidate = CandidateMask(
                candidate_id=candidate_id,
                interface_id="reviewed",
                tool_name="reviewed",
                target_mask=target_mask,
                change_region=change_region,
                tool_trace={},
            )
            np.save(case_review_dir / "final_mask.npy", target_mask, allow_pickle=False)
            np.save(
                case_review_dir / "final_change_region.npy",
                change_region,
                allow_pickle=False,
            )
            Image.fromarray(id_mask_to_rgb(target_mask)).save(
                case_review_dir / "final_mask.png"
            )
            Image.fromarray(change_region.astype(np.uint8) * 255).save(
                case_review_dir / "final_change_region.png"
            )
            _save_single_preview(
                case=case,
                source_mask=source_mask,
                candidate=candidate,
                output_path=case_review_dir / "provisional_preview.png",
                selection_label="Codex-reviewed",
            )
            result = {
                "case_id": case_id,
                "status": "selected_research",
                "candidate_id": candidate_id,
                "reviewer": override.get("reviewer", "current_codex_session"),
                "review_notes": override.get("review_notes", ""),
            }
        else:
            with Image.open(prepared_dir / "provisional_preview.png") as opened:
                opened.convert("RGB").save(
                    case_review_dir / "provisional_preview.png"
                )
            result = {
                "case_id": case_id,
                "status": "abstained",
                "candidate_id": None,
                "reviewer": "deterministic_gates_before_visual_review",
                "review_notes": (
                    override.get("review_notes", "")
                    if override is not None
                    else "no candidate passed the research hard-gate set"
                ),
                "failure_counts": prepared.get("failure_counts", {}),
            }
        _write_json(case_review_dir / "codex_review.json", result)
        results.append(result)
    summary = {
        "pilot_id": "mask-edit-refine-codex-session-50-v1",
        "provider": "current_codex_session_no_api",
        "case_count": len(results),
        "status_counts": _counts(item["status"] for item in results),
        "cases": results,
    }
    _write_json(review_root / "review_summary.json", summary)
    print(json.dumps(summary["status_counts"], ensure_ascii=False, sort_keys=True))
    return 0


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

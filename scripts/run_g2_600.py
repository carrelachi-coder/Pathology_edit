#!/usr/bin/env python3
"""Run G2-600 by scheduling the product manifest workflow and summarizing it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_mask_edit.audit.staged_review import (  # noqa: E402
    audit_target_mask,
    canonicalize_mask_stage_artifacts,
)
from phase3_mask_edit.audit.quality import QualityPolicy  # noqa: E402
from phase3_mask_edit.core.gland_region import (  # noqa: E402
    GLAS_WHOLE_GLAND_CELL_REGION_POLICY,
    SEMANTIC_CELL_DELETION_REGION_POLICY,
    SEMANTIC_NUCLEI_GENERATION_REGION_POLICY,
)
from scripts.build_g2_600_manifest import (  # noqa: E402
    build_product_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_RUNNER = REPO_ROOT / "scripts" / "run_phase3_manifest_pipeline.py"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mask = subparsers.add_parser("mask")
    mask.add_argument("--manifest", type=Path, required=True)
    mask.add_argument("--reserves", type=Path, required=True)
    mask.add_argument("--output", type=Path, required=True)
    mask.add_argument("--max-rounds", type=int, default=4)
    mask.add_argument("--seed", type=int, default=42)
    mask.add_argument("--expected-count", type=int, default=600)
    mask.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Restore accepted masks from completed mask_round_* artifacts and "
            "continue only unresolved replacement chains."
        ),
    )
    _add_product_options(mask)

    nuclei = subparsers.add_parser("nuclei")
    nuclei.add_argument("--manifest", type=Path, required=True)
    nuclei.add_argument("--approved-mask-manifest", type=Path, required=True)
    nuclei.add_argument("--output", type=Path, required=True)
    nuclei.add_argument("--expected-count", type=int, default=600)
    nuclei.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse successful nuclei results and rerun only failed or missing "
            "case ids through the same product runner."
        ),
    )
    _add_product_options(nuclei)

    image = subparsers.add_parser("image")
    image.add_argument("--manifest", type=Path, required=True)
    image.add_argument("--approved-mask-manifest", type=Path, required=True)
    image.add_argument("--approved-nuclei-manifest", type=Path, required=True)
    image.add_argument("--output", type=Path, required=True)
    image.add_argument("--expected-count", type=int, default=600)
    image.add_argument(
        "--gpu-ids",
        default="",
        help=(
            "Optional comma-separated physical GPU ids. Multiple ids shard "
            "the frozen cohort deterministically while every shard still "
            "uses the product manifest workflow."
        ),
    )
    image.add_argument("--max-repair-rounds", type=int, default=2)
    _add_product_options(image)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--expected-count", type=int, default=600)

    args = parser.parse_args(argv)
    if args.command == "mask":
        return run_mask_stage(args)
    if args.command == "nuclei":
        return run_nuclei_stage(args)
    if args.command == "image":
        return run_image_stage(args)
    return summarize_runs(
        args.run_root,
        args.output,
        expected_count=args.expected_count,
    )


def run_mask_stage(args: argparse.Namespace) -> int:
    source_manifest = _read_json(args.manifest)
    active_cases = list(source_manifest.get("cases") or [])
    if len(active_cases) != args.expected_count:
        raise ValueError(
            "G2 mask stage case count mismatch: "
            f"{len(active_cases)} != {args.expected_count}."
        )
    reserves = _read_csv(args.reserves)
    reserve_index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in reserves:
        reserve_index[(row["g2_organ"], row["g2_primitive"])].append(row)
    used_stems = {
        (str(case["dataset"]), str(case["sample_id"]))
        for case in active_cases
    }
    accepted_entries: list[dict[str, Any]] = []
    accepted_cases: list[dict[str, Any]] = []
    pending_cases = active_cases
    start_round = 1
    args.output.mkdir(parents=True, exist_ok=True)

    if args.resume_existing:
        restored = _restore_mask_round_history(
            args.output,
            current_cases=active_cases,
        )
        used_stems.update(restored["used_stems"])
        reconciliation = _reconcile_mask_cohort(
            accepted_entries=restored["accepted_entries"],
            accepted_cases=restored["accepted_cases"],
            unresolved_cases=restored["rejected_cases"],
            target_cases=active_cases,
            reserve_index=reserve_index,
            used_stems=used_stems,
            seed=args.seed,
            source_manifest_path=args.manifest,
            source_manifest=source_manifest,
        )
        accepted_entries.extend(reconciliation["accepted_entries"])
        accepted_cases.extend(reconciliation["accepted_cases"])
        pending_cases = reconciliation["pending_cases"]
        for entry in reconciliation["accepted_entries"]:
            _approve_stage_entry(
                entry,
                stage="mask",
                decision_source="g2_automatic_mask_gate_resume_revalidation",
            )
        for entry in reconciliation["superseded_entries"]:
            _supersede_mask_stage_entry(
                entry,
                decision_source="g2_global_wsi_reconciliation",
            )
        _write_json(
            args.output / "mask_resume_contract.json",
            {
                "schema_version": 1,
                "current_source_manifest": str(args.manifest),
                "current_source_manifest_sha256": _sha256_file(args.manifest),
                "historical_round_manifests": [
                    {
                        "path": str(path),
                        "sha256": _sha256_file(path),
                    }
                    for path in restored["round_manifests"]
                ],
                "current_contract_revalidation_count": restored[
                    "revalidation_count"
                ],
                "current_contract_revalidation_passed": restored[
                    "revalidation_passed"
                ],
                "retained_accepted_count": len(accepted_cases),
                "superseded_accepted_count": len(
                    reconciliation["superseded_entries"]
                ),
                "pending_replacement_count": len(pending_cases),
                "reconciliation_path": str(
                    args.output / "mask_resume_reconciliation.json"
                ),
                "next_round": restored["next_round"],
            },
        )
        _write_json(
            args.output / "mask_resume_reconciliation.json",
            reconciliation["provenance"],
        )
        start_round = int(restored["next_round"])
        if len(accepted_cases) + len(pending_cases) != args.expected_count:
            raise RuntimeError(
                "Restored G2 mask history does not preserve cohort cardinality: "
                f"{len(accepted_cases)} accepted + {len(pending_cases)} pending "
                f"!= {args.expected_count}."
            )

    round_index = start_round - 1
    resolved = not pending_cases
    round_numbers = (
        range(start_round, args.max_rounds + 1) if pending_cases else ()
    )
    for round_index in round_numbers:
        round_manifest = {
            **source_manifest,
            "cases": pending_cases,
            "g2_mask_round": round_index,
        }
        round_manifest_path = args.output / f"mask_round_{round_index:02d}.json"
        _write_json(round_manifest_path, round_manifest)
        run_id = f"mask_round_{round_index:02d}"
        _run_product(
            manifest=round_manifest_path,
            output_root=args.output,
            run_id=run_id,
            stop_after="mask",
            args=args,
            allow_case_failures=True,
        )
        batch_root = args.output / run_id
        summary = _read_json(batch_root / "batch_summary.json")
        case_by_id = {
            str(case["case_id"]): case for case in pending_cases
        }
        rejected: list[dict[str, Any]] = []
        round_accepted_entries: list[dict[str, Any]] = []
        round_accepted_cases: list[dict[str, Any]] = []
        for result in summary.get("results") or []:
            case_id = str(result.get("case_id") or "")
            case = case_by_id[case_id]
            mask_stage = result.get("mask_stage")
            if (
                result.get("status") == "completed"
                and isinstance(mask_stage, Mapping)
                and bool(mask_stage.get("audit_passed"))
            ):
                entry = {
                    "case_id": case_id,
                    "condition_id": case.get("condition_id"),
                    "dataset": case.get("dataset"),
                    "variant_id": result.get("variant_id"),
                    "run_dir": result.get("output_dir"),
                    **dict(mask_stage),
                }
                round_accepted_entries.append(entry)
                round_accepted_cases.append(case)
            else:
                rejected.append(case)
        if not rejected:
            for entry in round_accepted_entries:
                _approve_stage_entry(
                    entry,
                    stage="mask",
                    decision_source="g2_automatic_mask_gate",
                )
            accepted_entries.extend(round_accepted_entries)
            accepted_cases.extend(round_accepted_cases)
            resolved = True
            break
        reconciliation = _reconcile_mask_cohort(
            accepted_entries=accepted_entries + round_accepted_entries,
            accepted_cases=accepted_cases + round_accepted_cases,
            unresolved_cases=rejected,
            target_cases=active_cases,
            reserve_index=reserve_index,
            used_stems=used_stems,
            seed=args.seed,
            source_manifest_path=args.manifest,
            source_manifest=source_manifest,
        )
        accepted_entries = reconciliation["accepted_entries"]
        accepted_cases = reconciliation["accepted_cases"]
        pending_cases = reconciliation["pending_cases"]
        for entry in accepted_entries:
            _approve_stage_entry(
                entry,
                stage="mask",
                decision_source="g2_automatic_mask_gate_reconciliation",
            )
        for entry in reconciliation["superseded_entries"]:
            _supersede_mask_stage_entry(
                entry,
                decision_source="g2_global_wsi_reconciliation",
            )
    if not resolved:
        raise RuntimeError(
            f"G2 mask stage still has failures after {args.max_rounds} total "
            "rounds. Re-run with --resume-existing and a larger --max-rounds."
        )

    if (
        len(accepted_entries) != args.expected_count
        or len(accepted_cases) != args.expected_count
    ):
        raise RuntimeError(
            "G2 mask stage did not freeze the required number of accepted "
            f"target masks: expected {args.expected_count}."
        )
    frozen_manifest = {**source_manifest, "cases": accepted_cases}
    frozen_manifest["mask_stage"] = {
        "status": "hash_frozen",
        "entry_count": args.expected_count,
        "replacement_rounds": round_index,
    }
    frozen_manifest_path = args.output / "g2_600_frozen_product_manifest.json"
    approved_path = args.output / "approved_mask_stage_manifest.json"
    _write_json(frozen_manifest_path, frozen_manifest)
    approved = {
        "schema_version": 1,
        "stage": "mask",
        "approval": {
            "status": "approved",
            "decision_source": "g2_automatic_mask_gate",
            "required_entry_count": args.expected_count,
            "approved_entry_count": args.expected_count,
        },
        "entry_count": args.expected_count,
        "all_automatic_checks_passed": True,
        "frozen_target_mask_consumed": False,
        "entries": accepted_entries,
        "frozen_product_manifest": str(frozen_manifest_path),
        "frozen_product_manifest_sha256": _sha256_file(frozen_manifest_path),
    }
    _write_json(approved_path, approved)
    print(
        json.dumps(
            {
                "status": "mask_stage_frozen",
                "manifest": str(frozen_manifest_path),
                "approved_mask_manifest": str(approved_path),
                "rounds": round_index,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def run_nuclei_stage(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = _read_json(args.manifest)
    expected_case_ids = [
        str(case["case_id"]) for case in source_manifest.get("cases") or []
    ]
    if len(expected_case_ids) != args.expected_count:
        raise RuntimeError(
            "G2 nuclei manifest case count does not match the expected count: "
            f"{len(expected_case_ids)} != {args.expected_count}."
        )

    if args.resume_existing:
        result_by_case = _collect_nuclei_results(
            args.output,
            expected_case_ids=expected_case_ids,
        )
        retry_case_ids = [
            case_id
            for case_id in expected_case_ids
            if not _nuclei_result_passed(result_by_case.get(case_id))
        ]
        if retry_case_ids:
            run_id = _next_nuclei_repair_run_id(args.output)
            _run_product(
                manifest=args.manifest,
                output_root=args.output,
                run_id=run_id,
                stop_after="nuclei",
                args=args,
                approved_mask_manifest=args.approved_mask_manifest,
                case_ids=retry_case_ids,
                allow_case_failures=True,
            )
            repair_summary = _read_json(
                args.output / run_id / "batch_summary.json"
            )
            for result in repair_summary.get("results") or []:
                result_by_case[str(result.get("case_id") or "")] = result
        summary = {
            "results": [
                result_by_case[case_id]
                for case_id in expected_case_ids
                if case_id in result_by_case
            ]
        }
    else:
        run_id = "nuclei"
        _run_product(
            manifest=args.manifest,
            output_root=args.output,
            run_id=run_id,
            stop_after="nuclei",
            args=args,
            approved_mask_manifest=args.approved_mask_manifest,
        )
        summary = _read_json(args.output / run_id / "batch_summary.json")

    entries = []
    for result in summary.get("results") or []:
        stage = result.get("nuclei_stage")
        if not _nuclei_result_passed(result):
            raise RuntimeError(
                "G2 nuclei stage failed; do not silently relax the cohort: "
                f"{result.get('case_id')}"
            )
        entry = {
            "case_id": result.get("case_id"),
            "condition_id": result.get("case_id"),
            "dataset": result.get("dataset"),
            "variant_id": result.get("variant_id"),
            "run_dir": result.get("output_dir"),
            **dict(stage),
        }
        _approve_stage_entry(
            entry,
            stage="nuclei",
            decision_source="g2_automatic_nuclei_contract",
        )
        entries.append(entry)
    if len(entries) != args.expected_count:
        raise RuntimeError(
            "G2 nuclei stage did not complete the required number of cases: "
            f"{len(entries)} != {args.expected_count}."
        )
    approved_path = args.output / "approved_nuclei_stage_manifest.json"
    _write_json(
        approved_path,
        {
            "schema_version": 1,
            "stage": "nuclei",
            "approval": {
                "status": "approved",
                "decision_source": "g2_automatic_nuclei_contract",
                "required_entry_count": args.expected_count,
                "approved_entry_count": args.expected_count,
            },
            "entry_count": args.expected_count,
            "all_automatic_checks_passed": True,
            "image_generation_started": False,
            "entries": entries,
        },
    )
    print(json.dumps({"approved_nuclei_manifest": str(approved_path)}, indent=2))
    return 0


def _nuclei_result_passed(result: Mapping[str, Any] | None) -> bool:
    if not isinstance(result, Mapping):
        return False
    stage = result.get("nuclei_stage")
    cell = result.get("cell")
    gland_policy = (
        cell.get("gland_structure_policy")
        if isinstance(cell, Mapping)
        else None
    )
    if not isinstance(gland_policy, Mapping):
        return False
    applied = bool(gland_policy.get("applied"))
    expected_deletion = (
        GLAS_WHOLE_GLAND_CELL_REGION_POLICY
        if applied
        else SEMANTIC_CELL_DELETION_REGION_POLICY
    )
    expected_generation = (
        GLAS_WHOLE_GLAND_CELL_REGION_POLICY
        if applied
        else SEMANTIC_NUCLEI_GENERATION_REGION_POLICY
    )
    if gland_policy.get("cell_deletion_region_policy") != expected_deletion:
        return False
    if gland_policy.get("nuclei_generation_region_policy") != expected_generation:
        return False
    if applied and gland_policy.get("image_and_nuclei_region_equal") is not True:
        return False
    return bool(
        result.get("status") == "completed"
        and isinstance(stage, Mapping)
        and stage.get("audit_passed")
    )


def _collect_nuclei_results(
    output: Path,
    *,
    expected_case_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    expected = set(expected_case_ids)
    summaries = [output / "nuclei" / "batch_summary.json"]
    summaries.extend(
        sorted(output.glob("nuclei_repair_[0-9][0-9]/batch_summary.json"))
    )
    result_by_case: dict[str, Mapping[str, Any]] = {}
    for summary_path in summaries:
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        for result in summary.get("results") or []:
            case_id = str(result.get("case_id") or "")
            if case_id in expected:
                result_by_case[case_id] = result
    if not result_by_case:
        raise FileNotFoundError(
            f"No reusable nuclei batch results found under {output}."
        )
    return result_by_case


def _next_nuclei_repair_run_id(output: Path) -> str:
    for index in range(1, 100):
        run_id = f"nuclei_repair_{index:02d}"
        if not (output / run_id).exists():
            return run_id
    raise RuntimeError("No free nuclei repair run id remains.")


def run_image_stage(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = _read_json(args.manifest)
    case_ids = [
        str(case["case_id"]) for case in source_manifest.get("cases") or []
    ]
    if len(case_ids) != args.expected_count or len(set(case_ids)) != len(case_ids):
        raise ValueError(
            "G2 image stage requires the exact unique frozen cohort: "
            f"{len(case_ids)} != {args.expected_count}."
        )
    approved = _read_json(args.approved_nuclei_manifest)
    approved_ids = {
        str(entry.get("case_id") or "")
        for entry in approved.get("entries") or []
        if entry.get("approval") == "approved"
    }
    if (
        approved.get("all_automatic_checks_passed") is not True
        or approved_ids != set(case_ids)
    ):
        raise ValueError(
            "Approved nuclei manifest does not cover the exact frozen cohort."
        )

    gpu_ids = [
        value.strip()
        for value in str(getattr(args, "gpu_ids", "") or "").split(",")
        if value.strip()
    ]
    if len(gpu_ids) <= 1:
        _run_product(
            manifest=args.manifest,
            output_root=args.output,
            run_id="image",
            stop_after="image",
            args=args,
            approved_mask_manifest=args.approved_mask_manifest,
            approved_nuclei_manifest=args.approved_nuclei_manifest,
        )
        run_root = args.output / "image"
    else:
        run_root = args.output / "image"
        run_root.mkdir(parents=True, exist_ok=True)
        pending = list(case_ids)
        round_failures: list[str] = []
        max_repair_rounds = max(0, int(args.max_repair_rounds))
        for round_index in range(max_repair_rounds + 1):
            round_name = (
                "shard"
                if round_index == 0
                else f"repair_{round_index:02d}_shard"
            )
            round_failures.extend(
                _run_parallel_image_round(
                    manifest=args.manifest,
                    run_root=run_root,
                    log_root=args.output,
                    run_name=round_name,
                    case_ids=pending,
                    gpu_ids=gpu_ids,
                    args=args,
                    approved_mask_manifest=args.approved_mask_manifest,
                    approved_nuclei_manifest=args.approved_nuclei_manifest,
                )
            )
            image_results = _collect_image_results(run_root)
            pending = [
                case_id
                for case_id in case_ids
                if not _image_result_passed(image_results.get(case_id))
            ]
            if not pending:
                break
        if pending:
            raise RuntimeError(
                "G2 image stage still has unresolved cases after repair: "
                f"count={len(pending)}; case_ids={pending}; "
                f"shard_failures={round_failures}"
            )
    return summarize_runs(
        run_root,
        args.output / "summary",
        expected_count=args.expected_count,
    )


def _partition_case_ids(
    case_ids: Sequence[str],
    shard_count: int,
) -> list[list[str]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    return [list(case_ids[index::shard_count]) for index in range(shard_count)]


def _run_parallel_image_round(
    *,
    manifest: Path,
    run_root: Path,
    log_root: Path,
    run_name: str,
    case_ids: Sequence[str],
    gpu_ids: Sequence[str],
    args: argparse.Namespace,
    approved_mask_manifest: Path,
    approved_nuclei_manifest: Path,
) -> list[str]:
    shards = _partition_case_ids(case_ids, len(gpu_ids))
    processes: list[tuple[int, str, Path, Any, subprocess.Popen[Any]]] = []
    for shard_index, (gpu_id, shard_case_ids) in enumerate(zip(gpu_ids, shards)):
        if not shard_case_ids:
            continue
        run_id = f"{run_name}_{shard_index:02d}"
        log_path = log_root / f"image_{run_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = _product_command(
            manifest=manifest,
            output_root=run_root,
            run_id=run_id,
            stop_after="image",
            args=args,
            approved_mask_manifest=approved_mask_manifest,
            approved_nuclei_manifest=approved_nuclei_manifest,
            case_ids=shard_case_ids,
        )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_index, gpu_id, log_path, log_handle, process))
    failures: list[str] = []
    for shard_index, gpu_id, log_path, log_handle, process in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append(
                f"{run_name}_{shard_index:02d}@GPU{gpu_id}:"
                f"exit={return_code}:log={log_path}"
            )
    return failures


def _image_result_passed(result: Mapping[str, Any] | None) -> bool:
    if not isinstance(result, Mapping) or result.get("status") != "completed":
        return False
    output_dir = Path(str(result.get("output_dir") or ""))
    summary_path = output_dir / "agentic_generation" / "pipeline_summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    selected = summary.get("selected_attempt")
    return isinstance(selected, Mapping) and selected.get("attempt_index") is not None


def _collect_image_results(run_root: Path) -> dict[str, Mapping[str, Any]]:
    summary_candidates = [run_root / "batch_summary.json"]
    summary_candidates.extend(run_root.glob("*/batch_summary.json"))
    summaries = sorted(
        (path for path in summary_candidates if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    result_by_case: dict[str, Mapping[str, Any]] = {}
    for summary_path in summaries:
        for result in _read_json(summary_path).get("results") or []:
            case_id = str(result.get("case_id") or "")
            if case_id:
                result_by_case[case_id] = result
    return result_by_case


def summarize_runs(
    run_root: Path,
    output: Path,
    *,
    expected_count: int = 600,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    image_results = _collect_image_results(run_root)
    summaries = sorted(
        Path(str(result["output_dir"]))
        / "agentic_generation"
        / "pipeline_summary.json"
        for result in image_results.values()
        if _image_result_passed(result)
    )
    if not summaries:
        summaries = sorted(
            run_root.glob("**/agentic_generation/pipeline_summary.json")
        )
    for summary_path in summaries:
        workflow = _read_json(summary_path)
        case_root = summary_path.parent.parent
        run_config = _read_json(case_root / "run_config.json")
        case = _mapping(run_config.get("case"))
        base = {
            "case_id": case.get("case_id"),
            "organ": case.get("organ"),
            "dataset": case.get("dataset"),
            "primitive": case.get("g2_primitive"),
            "workflow_status": workflow.get("status"),
            "pipeline_summary": str(summary_path),
            "generation_report": str(
                summary_path.parent / "generation_report.json"
            ),
        }
        selected_index = _mapping(workflow.get("selected_attempt")).get(
            "attempt_index"
        )
        for attempt in workflow.get("attempts") or []:
            verification = _mapping(attempt.get("verification"))
            row = {
                **base,
                "attempt_index": attempt.get("attempt_index"),
                "model": attempt.get("requested_mode"),
                "selected": attempt.get("attempt_index") == selected_index,
                "passed": verification.get("passed"),
                "quality_score": verification.get("quality_score"),
                "evidence_coverage": verification.get("evidence_coverage"),
                "scientific_status": verification.get("scientific_status"),
                "failed_checks": json.dumps(
                    verification.get("failed_checks") or [],
                    ensure_ascii=False,
                ),
                "reason_codes": json.dumps(
                    verification.get("reason_codes") or [],
                    ensure_ascii=False,
                ),
                "image_path": _mapping(attempt.get("artifact")).get(
                    "image_path"
                ),
                "image_sha256": _optional_sha256(
                    _mapping(attempt.get("artifact")).get("image_path")
                ),
            }
            for name, value in _mapping(
                verification.get("component_scores")
            ).items():
                row[f"score_{name}"] = value
            for name, value in _mapping(verification.get("metrics")).items():
                if isinstance(value, (int, float)):
                    row[f"metric_{name}"] = value
            attempts.append(row)
            if row["selected"]:
                finals.append(dict(row))
    if len(finals) != expected_count:
        raise RuntimeError(
            "G2 image summary selected-final count mismatch: "
            f"{len(finals)} != {expected_count}."
        )
    _write_csv(output / "g2_600_attempts.csv", attempts)
    _write_csv(output / "g2_600_selected_final.csv", finals)
    grouped = _grouped_summary(finals)
    payload = {
        "schema_version": 1,
        "evaluator_policy_id": QualityPolicy().policy_id,
        "semantic_region_policy": "exact_source_target_tissue_difference",
        "preservation_exclusion_region_policy": (
            "full_generation_change_region"
        ),
        "attempt_count": len(attempts),
        "final_count": len(finals),
        "overall": grouped["overall"],
        "by_organ": grouped["by_organ"],
        "by_primitive": grouped["by_primitive"],
        "by_route": grouped["by_route"],
        "attempts_csv": str(output / "g2_600_attempts.csv"),
        "selected_final_csv": str(output / "g2_600_selected_final.csv"),
        "interpretation": (
            "validated rate is an automated engineering endpoint, not a "
            "clinical success rate"
        ),
    }
    _write_json(output / "g2_600_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _grouped_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(items)
        return {
            "count": count,
            "quality_score_mean": (
                sum(float(item.get("quality_score") or 0.0) for item in items)
                / count
                if count
                else None
            ),
            "engineering_validated_rate": (
                sum(
                    str(item.get("workflow_status"))
                    in {"validated_first_pass", "recovered"}
                    for item in items
                )
                / count
                if count
                else None
            ),
            "first_pass_rate": (
                sum(
                    str(item.get("workflow_status"))
                    == "validated_first_pass"
                    for item in items
                )
                / count
                if count
                else None
            ),
            "recovery_rate": (
                sum(str(item.get("workflow_status")) == "recovered" for item in items)
                / count
                if count
                else None
            ),
            "evaluator_uncertain_rate": (
                sum(
                    str(item.get("workflow_status")) == "evaluator_uncertain"
                    for item in items
                )
                / count
                if count
                else None
            ),
        }

    def group(field: str) -> dict[str, Any]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(field))].append(row)
        return {
            name: summarize(items)
            for name, items in sorted(grouped.items())
        }

    return {
        "overall": summarize(rows),
        "by_organ": group("organ"),
        "by_primitive": group("primitive"),
        "by_route": group("model"),
    }


def _run_product(
    *,
    manifest: Path,
    output_root: Path,
    run_id: str,
    stop_after: str,
    args: argparse.Namespace,
    approved_mask_manifest: Path | None = None,
    approved_nuclei_manifest: Path | None = None,
    case_ids: Sequence[str] = (),
    allow_case_failures: bool = False,
) -> None:
    command = _product_command(
        manifest=manifest,
        output_root=output_root,
        run_id=run_id,
        stop_after=stop_after,
        args=args,
        approved_mask_manifest=approved_mask_manifest,
        approved_nuclei_manifest=approved_nuclei_manifest,
        case_ids=case_ids,
    )
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode:
        batch_summary = output_root / run_id / "batch_summary.json"
        if allow_case_failures and batch_summary.is_file():
            return
        raise RuntimeError(
            f"Product manifest runner failed with code {result.returncode}."
        )


def _product_command(
    *,
    manifest: Path,
    output_root: Path,
    run_id: str,
    stop_after: str,
    args: argparse.Namespace,
    approved_mask_manifest: Path | None = None,
    approved_nuclei_manifest: Path | None = None,
    case_ids: Sequence[str] = (),
) -> list[str]:
    command = [
        sys.executable,
        str(PRODUCT_RUNNER),
        "--manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--stop-after",
        stop_after,
        "--variants",
        "instruction",
    ]
    if approved_mask_manifest is not None:
        command.extend(
            ["--approved-mask-manifest", str(approved_mask_manifest)]
        )
    if approved_nuclei_manifest is not None:
        command.extend(
            ["--approved-nuclei-manifest", str(approved_nuclei_manifest)]
        )
    for case_id in case_ids:
        command.extend(["--case-id", str(case_id)])
    for option in (
        "api_base_url",
        "api_key_env",
        "api_model",
        "contour_api_base_url",
        "contour_api_key_env",
        "contour_api_model",
    ):
        value = getattr(args, option, None)
        if value:
            command.extend([f"--{option.replace('_', '-')}", str(value)])
    return command


def _approve_stage_entry(
    entry: dict[str, Any],
    *,
    stage: str,
    decision_source: str,
) -> None:
    lock_path = Path(str(entry["lock_path"]))
    lock = _read_json(lock_path)
    if stage == "mask":
        digest = str(entry["target_tissue_sha256"])
        entry["approved_target_sha256"] = digest
        lock["approval"] = {
            "status": "approved",
            "approved_target_sha256": digest,
            "decision_source": decision_source,
        }
    else:
        digest = str(entry["target_nuclei_sha256"])
        entry["approved_target_nuclei_sha256"] = digest
        lock["approval"] = {
            "status": "approved",
            "approved_target_nuclei_sha256": digest,
            "decision_source": decision_source,
        }
    entry["approval"] = "approved"
    _write_json(lock_path, lock)


def _supersede_mask_stage_entry(
    entry: dict[str, Any],
    *,
    decision_source: str,
) -> None:
    lock_path = Path(str(entry["lock_path"]))
    lock = _read_json(lock_path)
    lock["approval"] = {
        "status": "superseded",
        "decision_source": decision_source,
        "reason": "released_to_satisfy_global_cell_and_wsi_constraints",
    }
    entry["approval"] = "superseded"
    _write_json(lock_path, lock)


def _restore_mask_round_history(
    output: Path,
    *,
    current_cases: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    round_manifests = sorted(output.glob("mask_round_[0-9][0-9].json"))
    if not round_manifests:
        raise FileNotFoundError(
            f"No completed G2 mask rounds found under {output}."
        )
    accepted_entries: list[dict[str, Any]] = []
    accepted_cases: list[dict[str, Any]] = []
    rejected_cases: list[dict[str, Any]] = []
    contract_rejected_by_id: dict[str, dict[str, Any]] = {}
    used_stems: set[tuple[str, str]] = set()
    seen_case_ids: set[str] = set()
    revalidation_count = 0
    revalidation_passed = 0
    current_case_by_id = {
        str(case["case_id"]): dict(case) for case in current_cases
    }

    for expected_round, manifest_path in enumerate(round_manifests, start=1):
        round_index = int(manifest_path.stem.rsplit("_", 1)[-1])
        if round_index != expected_round:
            raise RuntimeError(
                "G2 mask round history is not contiguous: expected round "
                f"{expected_round}, found {round_index}."
            )
        round_manifest = _read_json(manifest_path)
        historical_cases = list(round_manifest.get("cases") or [])
        cases = [
            {
                **dict(case),
                **current_case_by_id.get(str(case.get("case_id") or ""), {}),
            }
            for case in historical_cases
        ]
        summary_path = output / manifest_path.stem / "batch_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing batch summary for {manifest_path}: {summary_path}"
            )
        results = list(_read_json(summary_path).get("results") or [])
        if len(results) != len(cases):
            raise RuntimeError(
                "Cannot resume an incomplete G2 mask round: "
                f"{manifest_path.stem} has {len(results)}/{len(cases)} results."
            )
        case_by_id = {str(case["case_id"]): case for case in cases}
        result_ids = {str(result.get("case_id") or "") for result in results}
        if result_ids != set(case_by_id):
            raise RuntimeError(
                f"Case/result mismatch in completed {manifest_path.stem}."
            )
        duplicate_ids = seen_case_ids.intersection(case_by_id)
        if duplicate_ids:
            raise RuntimeError(
                "G2 replacement history reuses case ids: "
                + ", ".join(sorted(duplicate_ids))
            )
        seen_case_ids.update(case_by_id)
        for case in cases:
            used_stems.add(
                (str(case.get("dataset")), str(case.get("sample_id")))
            )

        rejected_cases = []
        for result in results:
            case_id = str(result.get("case_id") or "")
            case = case_by_id[case_id]
            mask_stage = result.get("mask_stage")
            if current_cases and result.get("status") == "completed":
                mask_stage = _revalidate_mask_stage(
                    result=result,
                    case=case,
                )
                revalidation_count += 1
                if isinstance(mask_stage, Mapping) and bool(
                    mask_stage.get("audit_passed")
                ):
                    revalidation_passed += 1
            if (
                result.get("status") == "completed"
                and isinstance(mask_stage, Mapping)
                and bool(mask_stage.get("audit_passed"))
            ):
                accepted_entries.append(
                    {
                        "case_id": case_id,
                        "condition_id": case.get("condition_id"),
                        "dataset": case.get("dataset"),
                        "variant_id": result.get("variant_id"),
                        "run_dir": result.get("output_dir"),
                        **dict(mask_stage),
                    }
                )
                accepted_cases.append(case)
            else:
                rejected_cases.append(case)
                if case_id in current_case_by_id:
                    contract_rejected_by_id[case_id] = dict(case)

    unresolved_by_id = {
        str(case.get("case_id") or ""): dict(case)
        for case in rejected_cases
    }
    unresolved_by_id.update(contract_rejected_by_id)

    return {
        "accepted_entries": accepted_entries,
        "accepted_cases": accepted_cases,
        "rejected_cases": [
            unresolved_by_id[case_id]
            for case_id in sorted(unresolved_by_id)
        ],
        "used_stems": used_stems,
        "next_round": len(round_manifests) + 1,
        "round_manifests": round_manifests,
        "revalidation_count": revalidation_count,
        "revalidation_passed": revalidation_passed,
    }


def _revalidate_mask_stage(
    *,
    result: Mapping[str, Any],
    case: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    mask_stage = result.get("mask_stage")
    if not isinstance(mask_stage, Mapping):
        return None
    source_path = Path(str(case.get("source_tissue_mask") or ""))
    target_path = Path(
        str(
            mask_stage.get("target_tissue_mask_path")
            or _mapping(result.get("tissue")).get("target_tissue_mask")
            or ""
        )
    )
    if not source_path.is_file() or not target_path.is_file():
        return mask_stage

    source = np.asarray(Image.open(source_path))
    target = np.asarray(Image.open(target_path))
    lock_path = Path(str(mask_stage.get("lock_path") or ""))
    lock = _read_json(lock_path) if lock_path.is_file() else {}
    change_path = Path(
        str(
            lock.get("change_region_path")
            or _mapping(result.get("tissue")).get("change_region")
            or ""
        )
    )
    cleanup: Mapping[str, Any] | None = None
    if change_path.is_file():
        target, _, cleanup = canonicalize_mask_stage_artifacts(
            source_mask=source,
            target_mask=target,
            target_mask_path=target_path,
            change_region_path=change_path,
            review_dir=target_path.parent / "stage_review",
        )
    audit = audit_target_mask(
        source_mask=source,
        target_mask=target,
        profile=str(case.get("profile") or case.get("dataset") or ""),
        case=case,
        phase3_info=_mapping(result.get("tissue")),
    )
    original_audit_path = str(mask_stage.get("audit_path") or "")
    review_dir = target_path.parent / "stage_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    audit_path = review_dir / "mask_audit_resume_revalidation.json"
    audit["resume_revalidation"] = {
        "source": "current_product_contract",
        "original_audit_path": original_audit_path,
    }
    if cleanup is not None:
        audit["target_mask_canonicalization"] = dict(cleanup)
    _write_json(audit_path, audit)
    target_sha256 = _sha256_file(target_path)
    if lock_path.is_file():
        asset_hashes = dict(_mapping(lock.get("asset_sha256")))
        asset_hashes["target_tissue"] = target_sha256
        if change_path.is_file():
            asset_hashes["change_region"] = _sha256_file(change_path)
        lock["asset_sha256"] = asset_hashes
        if cleanup is not None:
            lock["target_mask_canonicalization"] = dict(cleanup)
        lock["audit_path"] = str(audit_path)
        lock["audit_passed"] = bool(audit.get("passed"))
        _write_json(lock_path, lock)
    return {
        **dict(mask_stage),
        "audit_passed": bool(audit.get("passed")),
        "audit_path": str(audit_path),
        "original_audit_path": original_audit_path,
        "target_tissue_sha256": target_sha256,
        "target_mask_canonicalization": (
            dict(cleanup) if cleanup is not None else None
        ),
        "resume_revalidated": True,
    }


def _replacement_cases(
    rejected: Sequence[Mapping[str, Any]],
    *,
    reserve_index: Mapping[tuple[str, str], list[dict[str, str]]],
    used_stems: set[tuple[str, str]],
    used_wsi_counts: dict[tuple[str, str], int],
    seed: int,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    replacements: list[dict[str, Any]] = []
    allocated = _allocate_reserve_rows(
        rejected,
        reserve_index=reserve_index,
        used_stems=used_stems,
        used_wsi_counts=used_wsi_counts,
        source_manifest=source_manifest,
    )
    release_path = str(
        _mapping(source_manifest.get("runtime"))
        .get("verification", {})
        .get(
            "product_release",
            "benchmark_configs/releases/online_agent_product_v1.json",
        )
    )
    for case in rejected:
        key = (str(case["organ"]), str(case["g2_primitive"]))
        replacement = allocated[key].pop(0)
        used_stems.add(
            (str(replacement["dataset"]), str(replacement["stem"]))
        )
        wsi_key = (key[0], str(replacement["wsi"]))
        used_wsi_counts[wsi_key] = used_wsi_counts.get(wsi_key, 0) + 1
        replacement_manifest = build_product_manifest(
            [replacement],
            seed=seed,
            source_manifest=source_manifest_path,
            release_path=release_path,
        )
        replacement_case = replacement_manifest["cases"][0]
        replacement_case["replacement_for_case_id"] = case["case_id"]
        replacements.append(replacement_case)
    return replacements


def _reconcile_mask_cohort(
    *,
    accepted_entries: Sequence[dict[str, Any]],
    accepted_cases: Sequence[Mapping[str, Any]],
    unresolved_cases: Sequence[Mapping[str, Any]],
    target_cases: Sequence[Mapping[str, Any]],
    reserve_index: Mapping[tuple[str, str], list[dict[str, str]]],
    used_stems: set[tuple[str, str]],
    seed: int,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    allocation = _allocate_resume_cohort(
        accepted_cases=accepted_cases,
        target_cases=target_cases,
        reserve_index=reserve_index,
        used_stems=used_stems,
        source_manifest=source_manifest,
    )
    retained_ids = set(allocation["retained_case_ids"])
    entry_by_id = {str(entry["case_id"]): entry for entry in accepted_entries}
    retained_cases = [
        dict(case)
        for case in accepted_cases
        if str(case["case_id"]) in retained_ids
    ]
    retained_entries = [entry_by_id[str(case["case_id"])] for case in retained_cases]
    dropped_cases = [
        dict(case)
        for case in accepted_cases
        if str(case["case_id"]) not in retained_ids
    ]
    dropped_entries = [entry_by_id[str(case["case_id"])] for case in dropped_cases]

    displaced_by_key: dict[
        tuple[str, str], list[tuple[int, dict[str, Any]]]
    ] = defaultdict(list)
    for priority, cases in ((0, unresolved_cases), (1, dropped_cases)):
        for case in cases:
            displaced_by_key[_case_cell(case)].append(
                (priority, dict(case))
            )
    for slots in displaced_by_key.values():
        slots.sort(
            key=lambda item: (
                item[0],
                str(item[1].get("case_id") or ""),
            )
        )

    release_path = str(
        _mapping(source_manifest.get("runtime"))
        .get("verification", {})
        .get(
            "product_release",
            "benchmark_configs/releases/online_agent_product_v1.json",
        )
    )
    pending_cases: list[dict[str, Any]] = []
    selected_reserves: list[dict[str, Any]] = []
    assigned_displaced_ids: set[str] = set()
    for key in sorted(allocation["selected_reserves"]):
        rows = allocation["selected_reserves"][key]
        slots = [item[1] for item in displaced_by_key[key]]
        if len(rows) > len(slots):
            raise RuntimeError(
                f"G2 reconciliation slot mismatch for {key}: "
                f"{len(rows)} reserves > {len(slots)} displaced cases."
            )
        for replacement, displaced in zip(rows, slots[: len(rows)]):
            assigned_displaced_ids.add(str(displaced["case_id"]))
            used_stems.add(
                (str(replacement["dataset"]), str(replacement["stem"]))
            )
            replacement_manifest = build_product_manifest(
                [replacement],
                seed=seed,
                source_manifest=source_manifest_path,
                release_path=release_path,
            )
            replacement_case = replacement_manifest["cases"][0]
            replacement_case["replacement_for_case_id"] = displaced["case_id"]
            pending_cases.append(replacement_case)
            selected_reserves.append(
                {
                    "organ": key[0],
                    "g2_primitive": key[1],
                    "dataset": replacement["dataset"],
                    "sample_id": replacement["stem"],
                    "wsi": replacement["wsi"],
                    "reserve_rank": int(replacement.get("reserve_rank") or 0),
                    "replacement_for_case_id": displaced["case_id"],
                }
            )
    pending_cases.sort(key=lambda case: str(case["case_id"]))
    provenance = {
        "schema_version": 1,
        "policy": "minimum_accepted_displacement_global_milp",
        "target_count": len(target_cases),
        "candidate_accepted_count": len(accepted_cases),
        "retained_accepted_count": len(retained_cases),
        "superseded_accepted_count": len(dropped_cases),
        "unresolved_count": len(unresolved_cases),
        "selected_reserve_count": len(selected_reserves),
        "retained_case_ids": [str(case["case_id"]) for case in retained_cases],
        "superseded_case_ids": [str(case["case_id"]) for case in dropped_cases],
        "selected_reserves": selected_reserves,
        "unassigned_historical_displaced_case_ids": sorted(
            str(case["case_id"])
            for _, slots in displaced_by_key.items()
            for _, case in slots
            if str(case["case_id"]) not in assigned_displaced_ids
        ),
    }
    return {
        "accepted_entries": retained_entries,
        "accepted_cases": retained_cases,
        "superseded_entries": dropped_entries,
        "superseded_cases": dropped_cases,
        "pending_cases": pending_cases,
        "provenance": provenance,
    }


def _allocate_resume_cohort(
    *,
    accepted_cases: Sequence[Mapping[str, Any]],
    target_cases: Sequence[Mapping[str, Any]],
    reserve_index: Mapping[tuple[str, str], list[dict[str, str]]],
    used_stems: set[tuple[str, str]],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    requirements: dict[tuple[str, str], int] = defaultdict(int)
    for case in target_cases:
        requirements[_case_cell(case)] += 1
    if not requirements:
        return {"retained_case_ids": [], "selected_reserves": {}}

    variables: list[
        tuple[
            str,
            tuple[str, str],
            tuple[str, str],
            tuple[str, str],
            Mapping[str, Any],
        ]
    ] = []
    for case in accepted_cases:
        key = _case_cell(case)
        variables.append(
            (
                "accepted",
                key,
                (str(case["dataset"]), str(case["sample_id"])),
                (key[0], str(case["wsi"])),
                case,
            )
        )
    for key in sorted(requirements):
        for row in reserve_index.get(key, []):
            patch_key = (str(row["dataset"]), str(row["stem"]))
            if patch_key in used_stems:
                continue
            variables.append(
                (
                    "reserve",
                    key,
                    patch_key,
                    (key[0], str(row.get("wsi") or "")),
                    row,
                )
            )
    if not variables:
        raise RuntimeError("No accepted or reserve G2 candidates remain.")

    primitive_rows = {key: index for index, key in enumerate(sorted(requirements))}
    patch_keys = sorted({item[2] for item in variables})
    patch_rows = {
        key: len(primitive_rows) + index for index, key in enumerate(patch_keys)
    }
    wsi_keys = sorted({item[3] for item in variables})
    wsi_rows = {
        key: len(primitive_rows) + len(patch_rows) + index
        for index, key in enumerate(wsi_keys)
    }
    row_count = len(primitive_rows) + len(patch_rows) + len(wsi_rows)
    matrix = lil_matrix((row_count, len(variables)), dtype=np.float64)
    lower = np.full(row_count, -np.inf, dtype=np.float64)
    upper = np.full(row_count, np.inf, dtype=np.float64)
    for key, index in primitive_rows.items():
        lower[index] = requirements[key]
        upper[index] = requirements[key]
    for index in patch_rows.values():
        upper[index] = 1.0
    wsi_caps = _mapping(
        _mapping(source_manifest.get("selection_policy")).get("actual_wsi_caps")
    )
    for key, index in wsi_rows.items():
        cap = int(wsi_caps.get(key[0]) or 0)
        upper[index] = float(cap) if cap else np.inf
    for column, (_, key, patch_key, wsi_key, _) in enumerate(variables):
        matrix[primitive_rows[key], column] = 1.0
        matrix[patch_rows[patch_key], column] = 1.0
        matrix[wsi_rows[wsi_key], column] = 1.0

    objective = np.asarray(
        [
            float(index)
            if kind == "accepted"
            else 1_000_000_000_000.0
            + int(item.get("reserve_rank") or (index + 1)) * 1_000_000.0
            + index
            for index, (kind, _, _, _, item) in enumerate(variables)
        ],
        dtype=np.float64,
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "G2 global cohort reconciliation is infeasible under exact cell, "
            "unique-patch, and frozen WSI-cap constraints: "
            f"{result.message}"
        )

    retained_case_ids: list[str] = []
    selected_reserves: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for value, (kind, key, _, _, item) in zip(result.x, variables):
        if float(value) <= 0.5:
            continue
        if kind == "accepted":
            retained_case_ids.append(str(item["case_id"]))
        else:
            selected_reserves[key].append(dict(item))
    retained_case_ids.sort()
    for rows in selected_reserves.values():
        rows.sort(
            key=lambda row: (
                int(row.get("reserve_rank") or 0),
                str(row.get("dataset") or ""),
                str(row.get("stem") or ""),
            )
        )
    return {
        "retained_case_ids": retained_case_ids,
        "selected_reserves": dict(selected_reserves),
    }


def _case_cell(case: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(case.get("organ") or case.get("g2_organ") or ""),
        str(case.get("g2_primitive") or ""),
    )


def _allocate_reserve_rows(
    rejected: Sequence[Mapping[str, Any]],
    *,
    reserve_index: Mapping[tuple[str, str], list[dict[str, str]]],
    used_stems: set[tuple[str, str]],
    used_wsi_counts: Mapping[tuple[str, str], int],
    source_manifest: Mapping[str, Any],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    requirements: dict[tuple[str, str], int] = defaultdict(int)
    for case in rejected:
        requirements[(str(case["organ"]), str(case["g2_primitive"]))] += 1
    if not requirements:
        return {}

    wsi_caps = _mapping(
        _mapping(source_manifest.get("selection_policy")).get(
            "actual_wsi_caps"
        )
    )
    variables: list[
        tuple[tuple[str, str], tuple[str, str], tuple[str, str], dict[str, str]]
    ] = []
    for key in sorted(requirements):
        organ = key[0]
        cap = int(wsi_caps.get(organ) or 0)
        for row in reserve_index.get(key, []):
            patch_key = (str(row["dataset"]), str(row["stem"]))
            wsi_key = (organ, str(row.get("wsi") or ""))
            if patch_key in used_stems:
                continue
            if cap and used_wsi_counts.get(wsi_key, 0) >= cap:
                continue
            variables.append((key, patch_key, wsi_key, row))
    if not variables:
        raise RuntimeError("No unused G2 reserve candidates remain.")

    primitive_rows = {key: index for index, key in enumerate(sorted(requirements))}
    patch_keys = sorted({item[1] for item in variables})
    patch_rows = {
        key: len(primitive_rows) + index for index, key in enumerate(patch_keys)
    }
    wsi_keys = sorted({item[2] for item in variables})
    wsi_rows = {
        key: len(primitive_rows) + len(patch_rows) + index
        for index, key in enumerate(wsi_keys)
    }
    row_count = len(primitive_rows) + len(patch_rows) + len(wsi_rows)
    matrix = lil_matrix((row_count, len(variables)), dtype=np.float64)
    lower = np.full(row_count, -np.inf, dtype=np.float64)
    upper = np.full(row_count, np.inf, dtype=np.float64)

    for key, index in primitive_rows.items():
        lower[index] = requirements[key]
        upper[index] = requirements[key]
    for index in patch_rows.values():
        upper[index] = 1.0
    for key, index in wsi_rows.items():
        cap = int(wsi_caps.get(key[0]) or 0)
        upper[index] = (
            max(0, cap - int(used_wsi_counts.get(key, 0))) if cap else np.inf
        )
    for column, (key, patch_key, wsi_key, _) in enumerate(variables):
        matrix[primitive_rows[key], column] = 1.0
        matrix[patch_rows[patch_key], column] = 1.0
        matrix[wsi_rows[wsi_key], column] = 1.0

    objective = np.asarray(
        [
            int(item[3].get("reserve_rank") or (index + 1)) * 1_000_000
            + index
            for index, item in enumerate(variables)
        ],
        dtype=np.float64,
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "G2 reserve allocation is infeasible under same-cell, unique-patch, "
            f"and frozen WSI-cap constraints: {result.message}"
        )

    selected: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for value, (key, _, _, row) in zip(result.x, variables):
        if float(value) > 0.5:
            selected[key].append(row)
    for key, required in requirements.items():
        selected[key].sort(
            key=lambda row: (
                int(row.get("reserve_rank") or 0),
                str(row.get("dataset") or ""),
                str(row.get("stem") or ""),
            )
        )
        if len(selected[key]) != required:
            raise RuntimeError(
                f"G2 reserve allocation cardinality mismatch for {key}: "
                f"{len(selected[key])} != {required}."
            )
    return dict(selected)


def _wsi_counts(cases: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for case in cases:
        counts[(str(case.get("organ") or ""), str(case.get("wsi") or ""))] += 1
    return counts


def _add_product_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--api-model")
    parser.add_argument("--contour-api-base-url")
    parser.add_argument("--contour-api-key-env")
    parser.add_argument("--contour-api-model")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    return _sha256_file(path) if path.is_file() else None


if __name__ == "__main__":
    sys.exit(main())

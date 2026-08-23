#!/usr/bin/env python3
"""Run qualified PANDA candidates to five full-gate passes per evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.g2_v2_shadow import _materialize_joint_context
from phase3_joint_edit_refine.portfolio_authority import (
    canonical_metadata_sha256,
)
from phase3_joint_edit_refine.semantic_parser import (
    PreboundSemanticParser,
    bind_semantic_intent,
)
from phase3_mask_edit_refine.evidence import sha256_file

SCHEMA_VERSION = "panda-primitive-full-shadow-replay-v1"
PASS_STATUS = "selected_research"
COMPILED_REVIEW_STATUS = "compiled_pending_visual_review"


def _directory_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"runtime asset directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_qualification_records(path: Path) -> list[dict[str, Any]]:
    """Load either the materializer's JSON manifest or a legacy JSONL ledger."""

    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return [dict(item) for item in payload["cases"]]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError("qualification ledger must be a JSON manifest or JSONL records")


def _qualification_record_passed(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return record.get("status") == "executable_preflight_passed" or (
        record.get("execution_allowed") is True
        and record.get("decision_status") == "eligible"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _nested_ranker_evidence(value: Any) -> list[dict[str, Any]]:
    evidence = []
    if isinstance(value, dict):
        if "ranker" in value or "ranker_provenance" in value:
            evidence.append(
                {
                    "ranker": value.get("ranker"),
                    "ranker_provenance": value.get("ranker_provenance"),
                }
            )
        for child in value.values():
            evidence.extend(_nested_ranker_evidence(child))
    elif isinstance(value, list):
        for child in value:
            evidence.extend(_nested_ranker_evidence(child))
    return evidence


def _frozen_ranker_binding_passed(
    evidence: list[dict[str, Any]], *, checkpoint_sha256: str,
    instance_library_sha256: str,
) -> bool:
    bindings = []
    for item in evidence:
        provenance = item.get("ranker_provenance")
        if not isinstance(provenance, dict):
            continue
        role = provenance.get("role")
        if role == "no_new_placement" and provenance.get("probnet_used") is False:
            bindings.append(True)
            continue
        if role == "legal_template_anchor_ranking_only":
            bindings.append(
                provenance.get("checkpoint_sha256") == checkpoint_sha256
            )
            continue
        if provenance.get("checkpoint_sha256"):
            bindings.append(
                provenance.get("checkpoint_sha256") == checkpoint_sha256
                and provenance.get("instance_library_sha256")
                == instance_library_sha256
            )
    return bool(bindings and all(bindings))


def _execution_evidence(
    *, summary_path: Path, workflow_status: str | None,
    selected_candidate_id: str | None, checkpoint_sha256: str,
    instance_library_sha256: str,
) -> dict[str, Any]:
    result = {
        "selected_gate_report_passed": False,
        "selected_hard_gate_failure_ids": [],
        "selected_gate_check_count": 0,
        "ranker_evidence": [],
        "frozen_ranker_binding_passed": False,
        "gate_report": None,
        "candidates_manifest": None,
    }
    if (
        workflow_status not in {PASS_STATUS, COMPILED_REVIEW_STATUS}
        or not selected_candidate_id
        or not summary_path.is_file()
    ):
        return result
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, list) or len(summary) != 1:
        return result
    artifacts = summary[0].get("artifact_paths") or {}
    gate_path = Path(str(artifacts.get("joint_gate_reports.json") or ""))
    candidates_path = Path(str(artifacts.get("candidates.json") or ""))
    result["gate_report"] = str(gate_path) if gate_path.is_file() else None
    result["candidates_manifest"] = (
        str(candidates_path) if candidates_path.is_file() else None
    )
    if gate_path.is_file():
        reports = json.loads(gate_path.read_text(encoding="utf-8"))
        selected_report = next(
            (
                item
                for item in reports
                if item.get("candidate_id") == selected_candidate_id
            ),
            None,
        )
        if selected_report is not None:
            checks = list(selected_report.get("checks") or [])
            hard_failures = [
                str(item.get("check_id"))
                for item in checks
                if item.get("severity") == "hard" and not item.get("passed")
            ]
            result["selected_gate_report_passed"] = bool(
                selected_report.get("passed") and not hard_failures
            )
            result["selected_hard_gate_failure_ids"] = hard_failures
            result["selected_gate_check_count"] = len(checks)
    if candidates_path.is_file():
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        selected = next(
            (
                item
                for item in candidates
                if item.get("candidate_id") == selected_candidate_id
            ),
            None,
        )
        if selected is not None:
            result["ranker_evidence"] = _nested_ranker_evidence(
                selected.get("tool_trace") or {}
            )
    result["frozen_ranker_binding_passed"] = _frozen_ranker_binding_passed(
        result["ranker_evidence"],
        checkpoint_sha256=checkpoint_sha256,
        instance_library_sha256=instance_library_sha256,
    )
    return result


def _compiled_review_candidate(
    *, artifact_paths: dict[str, Any], abstain_reasons: list[str]
) -> str | None:
    """Return the offline critic's deterministic, hard-gate-passing choice."""

    if abstain_reasons != [
        "independent_mask_condition_critic_approval_required"
    ]:
        return None
    candidates_path = Path(str(artifact_paths.get("candidates.json") or ""))
    gates_path = Path(str(artifact_paths.get("joint_gate_reports.json") or ""))
    critic_path = Path(str(artifact_paths.get("joint_critic.json") or ""))
    if not all(path.is_file() for path in (candidates_path, gates_path, critic_path)):
        return None
    critic = json.loads(critic_path.read_text(encoding="utf-8"))
    rankings = critic.get("rankings") if isinstance(critic, dict) else None
    if not isinstance(rankings, list) or not rankings:
        return None
    ranking = rankings[0]
    if not isinstance(ranking, dict) or ranking.get("veto_reasons"):
        return None
    candidate_id = str(ranking.get("candidate_id") or "")
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    reports = json.loads(gates_path.read_text(encoding="utf-8"))
    candidate_exists = any(
        isinstance(item, dict) and item.get("candidate_id") == candidate_id
        for item in candidates
    )
    passing_report = next(
        (
            item
            for item in reports
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id
        ),
        None,
    )
    if not candidate_exists or not isinstance(passing_report, dict):
        return None
    failed_hard = [
        item
        for item in passing_report.get("checks", ())
        if isinstance(item, dict)
        and item.get("severity") == "hard"
        and item.get("passed") is not True
    ]
    if passing_report.get("passed") is not True or failed_hard:
        return None
    return candidate_id


def _run_case(
    *, case_id: str, manifest: Path, case_root: Path,
    checkpoint: Path, library: Path, checkpoint_sha256: str,
    instance_library_sha256: str, device: str, timeout_seconds: int,
) -> dict[str, Any]:
    case_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "phase3_joint_edit_refine.cli",
        "--manifest",
        str(manifest),
        "--output-root",
        str(case_root),
        "--case-id",
        case_id,
        "--agent-mode",
        "offline",
        "--semantic-parser",
        "prebound",
        "--cell-executor",
        "mature",
        "--probnet-checkpoint",
        str(checkpoint),
        "--nuclei-instance-library",
        str(library),
        "--probnet-dataset",
        "PANDA",
        "--device",
        device,
        "--meta-eval",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return_code = int(process.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return_code = 124
    duration = round(time.monotonic() - started, 3)
    stdout_path = case_root / "bounded_stdout.log"
    stderr_path = case_root / "bounded_stderr.log"
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")
    summary_path = case_root / "joint_run_summary.json"
    workflow_status = None
    selected_candidate_id = None
    abstain_reasons = []
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, list) and len(summary) == 1:
            workflow_status = summary[0].get("status")
            selected_candidate_id = summary[0].get("selected_candidate_id")
            abstain_reasons = list(summary[0].get("abstain_reasons") or [])
            if workflow_status == "review_required":
                selected_candidate_id = _compiled_review_candidate(
                    artifact_paths=dict(summary[0].get("artifact_paths") or {}),
                    abstain_reasons=abstain_reasons,
                )
                if selected_candidate_id is not None:
                    workflow_status = COMPILED_REVIEW_STATUS
    evidence = _execution_evidence(
        summary_path=summary_path,
        workflow_status=(str(workflow_status) if workflow_status else None),
        selected_candidate_id=(
            str(selected_candidate_id) if selected_candidate_id else None
        ),
        checkpoint_sha256=checkpoint_sha256,
        instance_library_sha256=instance_library_sha256,
    )
    full_pass = bool(
        not timed_out
        and return_code == 0
        and workflow_status in {PASS_STATUS, COMPILED_REVIEW_STATUS}
        and evidence["selected_gate_report_passed"]
        and evidence["frozen_ranker_binding_passed"]
    )
    return {
        "case_id": case_id,
        "full_gate_passed": full_pass,
        "workflow_status": workflow_status,
        "selected_candidate_id": selected_candidate_id,
        "abstain_reasons": abstain_reasons,
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "wall_time_seconds": duration,
        "joint_run_summary": str(summary_path) if summary_path.is_file() else None,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        **evidence,
        "llm_api_used": False,
        "llm_h_e_exposure": False,
    }


def _select_diverse_passes(
    records: list[dict[str, Any]], *, policy: dict[str, Any]
) -> list[dict[str, Any]]:
    final_count = int(policy["final_case_count"])
    minimum_slides = int(policy["minimum_distinct_source_slides"])
    maximum_per_slide = int(policy["maximum_cases_per_source_slide"])
    passed = sorted(
        (item for item in records if item.get("full_gate_passed")),
        key=lambda value: (
            int(value["candidate_pool_rank"]),
            str(value["case_id"]),
        ),
    )
    best: tuple[tuple[int, ...], tuple[str, ...], tuple[dict[str, Any], ...]] | None = None
    for selected in combinations(passed, final_count):
        samples = [str(item["source_sample_id"]) for item in selected]
        if len(samples) != len(set(samples)):
            continue
        slide_counts = Counter(str(item["source_slide_id"]) for item in selected)
        if (
            len(slide_counts) < minimum_slides
            or max(slide_counts.values(), default=0) > maximum_per_slide
        ):
            continue
        key = (
            tuple(int(item["candidate_pool_rank"]) for item in selected),
            tuple(str(item["case_id"]) for item in selected),
            selected,
        )
        if best is None or key[:2] < best[:2]:
            best = key
    return list(best[2]) if best is not None else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-manifest", type=Path, required=True)
    parser.add_argument("--qualification-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probnet-checkpoint", type=Path, required=True)
    parser.add_argument("--nuclei-library", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--evaluation-indices",
        help="optional comma-separated evaluation indices for a replay shard",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    authority = json.loads(args.authority_manifest.read_text(encoding="utf-8"))
    if authority.get("schema_version") != "panda-primitive-shadow-authority-v1":
        raise ValueError("unsupported PANDA authority manifest")
    declared_authority_digest = authority.get("authority_manifest_sha256")
    unsigned_authority = dict(authority)
    unsigned_authority.pop("authority_manifest_sha256", None)
    if declared_authority_digest != canonical_metadata_sha256(unsigned_authority):
        raise ValueError("PANDA authority manifest digest mismatch")
    runtime = authority["runtime_authority"]
    code_inventory = runtime.get("runtime_code_files")
    if (
        not isinstance(code_inventory, list)
        or canonical_metadata_sha256(code_inventory)
        != runtime.get("runtime_code_inventory_sha256")
    ):
        raise ValueError("runtime code inventory authority is absent or malformed")
    for item in code_inventory:
        relative = Path(str(item.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("runtime code inventory contains an unsafe path")
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"runtime code drifted: {relative}")
    checkpoint_digest = sha256_file(args.probnet_checkpoint)
    if checkpoint_digest != runtime["mature_probnet_executor"]["sha256"]:
        raise ValueError("Mature ProbNet checkpoint drifted")
    if checkpoint_digest != runtime["frozen_probnet_spatial_ranker"]["sha256"]:
        raise ValueError("Frozen ProbNet ranker checkpoint drifted")
    if not runtime.get("executor_ranker_same_checkpoint"):
        raise ValueError("executor/ranker shared checkpoint binding is absent")
    instance_library_digest = _directory_sha256(args.nuclei_library)
    if instance_library_digest != runtime[
        "panda_nucleus_instance_library"
    ]["sha256"]:
        raise ValueError("PANDA nucleus instance library drifted")
    skill_catalog = (
        REPOSITORY_ROOT
        / "phase3_joint_edit_refine"
        / "skills"
        / "catalog"
    )
    if _directory_sha256(skill_catalog) != runtime.get(
        "joint_skill_catalog", {}
    ).get("sha256"):
        raise ValueError("joint skill catalog drifted")
    qualification_manifest = Path(
        authority["candidate_qualification_manifest"]
    )
    if sha256_file(qualification_manifest) != authority[
        "candidate_qualification_manifest_sha256"
    ]:
        raise ValueError("candidate qualification manifest drifted")
    qualification_payload = json.loads(
        qualification_manifest.read_text(encoding="utf-8")
    )
    qualification = {
        str(item["case_id"]): item
        for item in _load_qualification_records(args.qualification_ledger)
    }
    qualified_rows = []
    for row in qualification_payload["cases"]:
        record = qualification.get(str(row["case_id"]))
        if _qualification_record_passed(record):
            qualified_rows.append(row)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in qualified_rows:
        grouped[int(row["evaluation_index"])].append(row)
    expected_evaluations = int(authority["evaluation_count"])
    requested_indices = (
        set(range(expected_evaluations))
        if not args.evaluation_indices
        else {
            int(value.strip())
            for value in args.evaluation_indices.split(",")
            if value.strip()
        }
    )
    if not requested_indices or not requested_indices.issubset(
        set(range(expected_evaluations))
    ):
        raise ValueError("evaluation indices are empty or outside authority range")
    diversity_by_evaluation = {
        int(item["evaluation_index"]): dict(item["final_diversity_policy"])
        for item in authority["evaluations"]
    }
    if set(grouped) != requested_indices:
        raise ValueError("one or more requested evaluations have no compiled candidates")
    short = [
        index for index in sorted(requested_indices) if len(grouped[index]) < 5
    ]
    if short:
        raise ValueError(
            "evaluations have fewer than five compiled candidates: "
            + ", ".join(map(str, short))
        )
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_digest = sha256_file(qualification_manifest)
    contexts = []
    rows_by_case = {}
    for row in qualified_rows:
        context = _materialize_joint_context(
            row, manifest_sha256=manifest_digest
        )
        context["provenance"] = {
            **context["provenance"],
            "shadow_runtime_authority_sha256": runtime[
                "runtime_authority_sha256"
            ],
            "mature_probnet_checkpoint_sha256": checkpoint_digest,
            "frozen_spatial_ranker_sha256": checkpoint_digest,
            "probnet_executor_ranker_same_checkpoint": True,
            "candidate_qualification_status": "executable_preflight_passed",
            "candidate_qualification_ledger_sha256": sha256_file(
                args.qualification_ledger
            ),
            "llm_api_used": False,
            "llm_h_e_exposure": False,
        }
        bound, _semantic = bind_semantic_intent(
            context,
            PreboundSemanticParser(context["prebound_semantic_intent"]),
        )
        bound.validate_local_inputs()
        contexts.append(context)
        rows_by_case[str(row["case_id"])] = row
    context_manifest = root / "qualified_joint_contexts.json"
    _write_json(context_manifest, contexts)
    context_manifest_sha256 = sha256_file(context_manifest)
    ledger_path = root / "full_execution_ledger.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and ledger_path.is_file():
        for item in _load_jsonl(ledger_path):
            unsigned = dict(item)
            declared = unsigned.pop("record_sha256", None)
            case_id = str(item.get("case_id") or "")
            if (
                case_id in rows_by_case
                and item.get("schema_version") == SCHEMA_VERSION
                and declared == canonical_metadata_sha256(unsigned)
                and item.get("runtime_authority_sha256")
                == runtime["runtime_authority_sha256"]
                and item.get("qualified_joint_contexts_sha256")
                == context_manifest_sha256
                and item.get("candidate_qualification_status")
                == "executable_preflight_passed"
            ):
                existing[case_id] = item
    selected_by_evaluation: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def refresh_selection(index: int) -> None:
        selected_by_evaluation[index] = _select_diverse_passes(
            [
                item
                for item in existing.values()
                if int(item["evaluation_index"]) == index
            ],
            policy=diversity_by_evaluation[index],
        )
    for index in sorted(requested_indices):
        refresh_selection(index)
    for evaluation_index in sorted(requested_indices):
        selected_by_evaluation[evaluation_index].sort(
            key=lambda item: int(item["candidate_pool_rank"])
        )
        if len(selected_by_evaluation[evaluation_index]) >= 5:
            continue
        rows = sorted(
            grouped[evaluation_index],
            key=lambda item: int(item["candidate_pool_rank"]),
        )
        for row in rows:
            if len(selected_by_evaluation[evaluation_index]) >= 5:
                break
            case_id = str(row["case_id"])
            if case_id in existing:
                continue
            print(
                json.dumps(
                    {
                        "stage": "full_joint_execution_and_gates",
                        "evaluation_index": evaluation_index,
                        "case_id": case_id,
                        "candidate_pool_rank": row["candidate_pool_rank"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            execution = _run_case(
                case_id=case_id,
                manifest=context_manifest,
                case_root=root / "cases" / case_id,
                checkpoint=args.probnet_checkpoint.resolve(),
                library=args.nuclei_library.resolve(),
                checkpoint_sha256=checkpoint_digest,
                instance_library_sha256=instance_library_digest,
                device=args.device,
                timeout_seconds=args.timeout_seconds,
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "evaluation_index": evaluation_index,
                "evaluation_id": row["evaluation_id"],
                "mechanism_id": row["mechanism_id"],
                "primitive_id": row["primitive_id"],
                "instruction": row["instruction"],
                "candidate_pool_rank": row["candidate_pool_rank"],
                "source_slide_id": row["source_slide_id"],
                "source_sample_id": row["source_sample_id"],
                "candidate_qualification_status": (
                    "executable_preflight_passed"
                ),
                "runtime_authority_sha256": runtime[
                    "runtime_authority_sha256"
                ],
                "qualified_joint_contexts_sha256": context_manifest_sha256,
                **execution,
            }
            record["record_sha256"] = canonical_metadata_sha256(record)
            existing[case_id] = record
            refresh_selection(evaluation_index)
            _write_jsonl(
                ledger_path,
                sorted(
                    existing.values(),
                    key=lambda item: (
                        int(item["evaluation_index"]),
                        int(item["candidate_pool_rank"]),
                    ),
                ),
            )
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    freeze_complete = all(
        len(selected_by_evaluation[index]) >= 5
        for index in requested_indices
    )
    frozen_evaluations = []
    if freeze_complete:
        for index in sorted(requested_indices):
            selected = selected_by_evaluation[index][:5]
            slides = [str(item["source_slide_id"]) for item in selected]
            samples = [str(item["source_sample_id"]) for item in selected]
            policy = diversity_by_evaluation[index]
            if (
                len(samples) != len(set(samples))
                or len(set(slides)) < int(policy["minimum_distinct_source_slides"])
                or max(Counter(slides).values())
                > int(policy["maximum_cases_per_source_slide"])
            ):
                raise ValueError("final selection violates source diversity policy")
            frozen_evaluations.append(
                {
                    "evaluation_index": index,
                    "evaluation_id": selected[0]["evaluation_id"],
                    "mechanism_id": selected[0]["mechanism_id"],
                    "primitive_id": selected[0]["primitive_id"],
                    "instruction": selected[0]["instruction"],
                    "frozen_case_count": 5,
                    "final_diversity_policy": policy,
                    "frozen_cases": [
                        {
                            "case_id": item["case_id"],
                            "candidate_pool_rank": item["candidate_pool_rank"],
                            "source_slide_id": item["source_slide_id"],
                            "source_sample_id": item["source_sample_id"],
                            "selected_candidate_id": item[
                                "selected_candidate_id"
                            ],
                            "record_sha256": item["record_sha256"],
                            "joint_run_summary": item["joint_run_summary"],
                            "gate_report": item["gate_report"],
                        }
                        for item in selected
                    ],
                }
            )
    all_records = sorted(
        existing.values(),
        key=lambda item: (
            int(item["evaluation_index"]),
            int(item["candidate_pool_rank"]),
        ),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "freeze_status": (
            (
                "frozen_complete_authority_candidate_execution_gates_passed"
                if requested_indices == set(range(expected_evaluations))
                else "frozen_complete_targeted_candidate_execution_gates_passed"
            )
            if freeze_complete
            else "incomplete_not_frozen"
        ),
        "authority_manifest": str(args.authority_manifest.resolve()),
        "authority_manifest_sha256": sha256_file(args.authority_manifest),
        "qualification_ledger": str(args.qualification_ledger.resolve()),
        "qualification_ledger_sha256": sha256_file(args.qualification_ledger),
        "qualified_joint_contexts": str(context_manifest),
        "qualified_joint_contexts_sha256": context_manifest_sha256,
        "full_execution_ledger": str(ledger_path),
        "full_execution_ledger_sha256": (
            sha256_file(ledger_path) if ledger_path.is_file() else None
        ),
        "evaluation_count": expected_evaluations,
        "requested_evaluation_indices": sorted(requested_indices),
        "requested_evaluation_count": len(requested_indices),
        "frozen_case_count": sum(
            len(item["frozen_cases"]) for item in frozen_evaluations
        ),
        "full_execution_attempt_count": len(all_records),
        "full_gate_pass_count": sum(
            bool(item["full_gate_passed"]) for item in all_records
        ),
        "workflow_status_counts": dict(
            sorted(Counter(str(item["workflow_status"]) for item in all_records).items())
        ),
        "runtime_authority": runtime,
        "frozen_evaluations": frozen_evaluations,
        "llm_api_used": False,
        "llm_h_e_exposure": False,
        "source_h_e_semantic_interpreter": "frozen_cellvit_only",
    }
    summary["freeze_manifest_sha256"] = canonical_metadata_sha256(summary)
    summary_path = root / "frozen_shadow_replay_manifest.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "freeze_status": summary["freeze_status"],
                "frozen_case_count": summary["frozen_case_count"],
                "full_execution_attempt_count": len(all_records),
                "summary": str(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if freeze_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

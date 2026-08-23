#!/usr/bin/env python3
"""Run the frozen Online Agent over five reviewed masks per open primitive."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


ORGAN_PROFILES = {
    "lung": "IGNITE",
    "oral": "ORCA",
    "skin": "PUMA",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agentic-python", type=Path, required=True)
    parser.add_argument("--agentic-runner", type=Path, required=True)
    parser.add_argument("--segmentator-python", type=Path, required=True)
    parser.add_argument("--pix2pix-checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,3,4,5,6,7")
    parser.add_argument("--cases-per-primitive", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _open_primitives_by_organ(audit_path: Path) -> dict[str, list[str]]:
    audit = _read_json(audit_path)
    return {
        str(organ["organ"]): list(organ["executable_unique_primitives"])
        for organ in audit["organs"]
        if organ["organ"] in ORGAN_PROFILES
    }


def _load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    open_primitives = _open_primitives_by_organ(args.audit)
    records: list[dict[str, Any]] = []
    for organ, profile in ORGAN_PROFILES.items():
        source_manifest = args.review_root / organ / "review_manifest.json"
        reviews = _read_json(source_manifest)["reviews"]
        expected = set(open_primitives[organ])
        missing = sorted(expected - set(reviews))
        if missing:
            raise RuntimeError(f"{organ} review manifest misses: {missing}")
        for primitive_id in sorted(expected):
            cases = reviews[primitive_id]["cases"]
            if len(cases) < args.cases_per_primitive:
                raise RuntimeError(
                    f"{organ}/{primitive_id} has only {len(cases)} cases"
                )
            for case_index, raw in enumerate(
                cases[: args.cases_per_primitive], start=1
            ):
                source_tissue = _load_mask(raw["source_tissue_mask"])
                target_tissue = _load_mask(raw["target_tissue_mask"])
                source_nuclei = _load_mask(raw["source_nuclei_mask"])
                target_nuclei = _load_mask(raw["target_nuclei_mask"])
                exact_joint = (source_tissue != target_tissue) | (
                    source_nuclei != target_nuclei
                )
                supplied_joint = _load_mask(raw["joint_change_mask"]) > 0
                if np.any(exact_joint & ~supplied_joint):
                    raise RuntimeError(
                        f"{organ}/{primitive_id}/{raw['case_id']}: "
                        "joint change misses final tissue+nuclei differences"
                    )
                case_root = (
                    args.output
                    / "cases"
                    / organ
                    / primitive_id
                    / raw["case_id"]
                )
                case_root.mkdir(parents=True, exist_ok=True)
                generation_candidate = (
                    case_root / "generation_candidate_full.png"
                )
                semantic_change_region = (
                    case_root / "semantic_change_exact.png"
                )
                Image.fromarray(exact_joint.astype(np.uint8) * 255).save(
                    semantic_change_region
                )
                Image.fromarray(
                    np.full(exact_joint.shape, 255, dtype=np.uint8)
                ).save(generation_candidate)
                records.append(
                    {
                        **raw,
                        "organ": organ,
                        "profile": profile,
                        "category": organ,
                        "primitive_id": primitive_id,
                        "instruction": primitive_id,
                        "case_index": case_index,
                        "mask_edit_joint_change_mask": raw[
                            "joint_change_mask"
                        ],
                        "semantic_change_region": str(
                            semantic_change_region.resolve()
                        ),
                        "generation_candidate": str(
                            generation_candidate.resolve()
                        ),
                        "agentic_generation_dir": str(
                            (case_root / "agentic_generation").resolve()
                        ),
                    }
                )
    manifest = {
        "schema_version": "lung-oral-skin-online-generation-batch-v1",
        "cases_per_primitive": args.cases_per_primitive,
        "organ_primitive_counts": {
            organ: len(open_primitives[organ]) for organ in ORGAN_PROFILES
        },
        "case_count": len(records),
        "records": records,
    }
    _write_json(args.output / "generation_manifest.json", manifest)
    return manifest


def _is_complete(output: Path) -> bool:
    workflow_path = output / "agentic_workflow.json"
    if not workflow_path.is_file() or not (output / "generated_image.png").is_file():
        return False
    workflow = _read_json(workflow_path)
    return not any(
        str(attempt.get("error") or "").startswith("generation failed:")
        for attempt in workflow.get("attempts", ())
    )


def _command(
    args: argparse.Namespace,
    record: dict[str, Any],
    gpu: int,
) -> list[str]:
    return [
        str(args.agentic_python),
        str(args.agentic_runner),
        "--profile",
        record["profile"],
        "--reference-image",
        record["source_image"],
        "--reference-tissue-mask",
        record["source_tissue_mask"],
        "--reference-nuclei-mask",
        record["source_nuclei_mask"],
        "--target-tissue-mask",
        record["target_tissue_mask"],
        "--target-nuclei-mask",
        record["target_nuclei_mask"],
        "--semantic-change-region",
        record["semantic_change_region"],
        "--generation-change-region",
        record["generation_candidate"],
        "--primitive-id",
        record["primitive_id"],
        "--output",
        record["agentic_generation_dir"],
        "--device",
        f"cuda:{gpu}",
        "--segmentator-python",
        str(args.segmentator_python),
        "--segmentator-device",
        f"cuda:{gpu}",
        "--cellvit-gpu",
        str(gpu),
        "--seed",
        str(args.seed),
        "--max-attempts",
        str(args.max_attempts),
    ]


def run_batch(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    buckets = [manifest["records"][index:: len(gpus)] for index in range(len(gpus))]
    events_path = args.output / "batch_events.jsonl"
    event_lock = threading.Lock()
    results: list[dict[str, Any]] = []

    def record_event(payload: dict[str, Any]) -> None:
        with event_lock:
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def worker(gpu: int, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        worker_results = []
        env = os.environ.copy()
        env["PATHOLOGY_PIX2PIX_CHECKPOINT"] = str(
            args.pix2pix_checkpoint.resolve()
        )
        for record in records:
            output = Path(record["agentic_generation_dir"])
            output.mkdir(parents=True, exist_ok=True)
            if _is_complete(output):
                result = {
                    "case_id": record["case_id"],
                    "organ": record["organ"],
                    "primitive_id": record["primitive_id"],
                    "gpu": gpu,
                    "status": "reused",
                    "returncode": 0,
                }
                record_event(result)
                worker_results.append(result)
                continue
            command = _command(args, record, gpu)
            log_path = output / "batch_launcher.log"
            record_event(
                {
                    "case_id": record["case_id"],
                    "organ": record["organ"],
                    "primitive_id": record["primitive_id"],
                    "gpu": gpu,
                    "status": "started",
                }
            )
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=args.agentic_runner.resolve().parents[1],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            workflow_path = output / "agentic_workflow.json"
            workflow_status = None
            if workflow_path.is_file():
                workflow_status = _read_json(workflow_path).get("status")
            result = {
                "case_id": record["case_id"],
                "organ": record["organ"],
                "primitive_id": record["primitive_id"],
                "gpu": gpu,
                "status": "finished",
                "returncode": completed.returncode,
                "workflow_status": workflow_status,
                "generated_image_exists": (
                    output / "generated_image.png"
                ).is_file(),
                "log": str(log_path),
            }
            record_event(result)
            worker_results.append(result)
        return worker_results

    events_path.unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, bucket)
            for gpu, bucket in zip(gpus, buckets)
        ]
        for future in futures:
            results.extend(future.result())
    results.sort(key=lambda item: (item["organ"], item["primitive_id"], item["case_id"]))
    _write_json(args.output / "batch_results.json", {"records": results})
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "case_count": manifest["case_count"]}))
        return 0
    results = run_batch(args, manifest)
    generated = sum(bool(item.get("generated_image_exists")) for item in results)
    print(
        json.dumps(
            {
                "status": "finished",
                "case_count": len(results),
                "generated_image_count": generated,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0 if generated == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

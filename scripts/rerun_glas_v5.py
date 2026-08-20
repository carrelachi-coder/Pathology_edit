#!/usr/bin/env python3
"""Recompile existing selected GLaS cases with new budget parameters."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_glas_primitive_mask_review import (
    EVALUATIONS,
    MASK_REVIEW_CELL_BUDGETS,
    _build_case,
    _run_case,
    _render_board,
    _write_json,
    _cross_meta_targets,
)
from phase3_mask_edit_refine.evidence import sha256_file

CROSS_META = Path("/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_val.json")
PROBNET = Path("/home/lyw/wqx-DL/flow-edit/hf_generation_release/pathology-probnet/best.pt")
LIBRARY = Path("/home/lyw/wqx-DL/flow-edit/FlowEdit-main/nuclei_library/GlaS")


def load_cross_meta():
    targets = _cross_meta_targets(CROSS_META)
    lookup = {}
    for r in targets:
        sid = r["sample_id"]
        if sid not in lookup:
            lookup[sid] = r
    return lookup


def _run_single(args_tuple):
    case_info, output_dir, lookup, timeout_seconds, device = args_tuple
    prim = case_info["primitive_id"]
    sid = case_info["sample_id"]
    row = lookup.get(sid)
    if not row:
        return None, {"primitive_id": prim, "sample_id": sid, "status": "skipped", "reason": "not found in cross-meta"}

    evaluation = next(e for e in EVALUATIONS if e.primitive_id == prim)

    payload = _build_case(
        evaluation,
        row,
        output_root=output_dir,
        native=None,
        attempt_index=case_info.get("compiler_seed", 0) or 1,
        portfolio_index=case_info.get("portfolio_index", 0),
        removal_variant=case_info.get("removal_variant", 0),
    )

    result = _run_case(
        payload,
        output_root=output_dir,
        checkpoint=PROBNET,
        library=LIBRARY,
        timeout_seconds=timeout_seconds,
        threads=4,
        device=device,
    )

    status = result["status"]
    reasons = result.get("abstain_reasons", [])
    print(f"{prim} | {sid} | {status}", flush=True)

    attempt = {
        "primitive_id": prim,
        "sample_id": sid,
        "status": status,
        "return_code": result["return_code"],
        "duration_seconds": result["duration_seconds"],
        "abstain_reasons": reasons,
        "portfolio_index": case_info.get("portfolio_index", 0),
        "removal_variant": case_info.get("removal_variant", 0),
    }
    if status in {"selected_research", "compiled_pending_visual_review"}:
        return {"row": row, "payload": payload, "run": result}, attempt
    return None, attempt


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    lookup = load_cross_meta()
    cases = json.loads(args.cases.read_text())

    selected_by_primitive = {}
    attempts_log = []

    work_items = [
        (c, args.output_dir, lookup, args.timeout, args.device)
        for c in cases
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_single, item): item for item in work_items}
        for future in as_completed(futures):
            selected, attempt = future.result()
            attempts_log.append(attempt)
            if selected is not None:
                prim = attempt["primitive_id"]
                if prim not in selected_by_primitive:
                    selected_by_primitive[prim] = []
                selected_by_primitive[prim].append(selected)

    boards = {}
    selected_manifest = []
    for evaluation in EVALUATIONS:
        prim = evaluation.primitive_id
        selected = selected_by_primitive.get(prim, [])
        print(f"{prim}: {len(selected)} selected", flush=True)
        if len(selected) >= 1:
            board = args.output_dir / "boards" / f"{prim}.png"
            records = _render_board(
                evaluation, selected, output_path=board, tile_size=384
            )
            boards[prim] = str(board)
            selected_manifest.extend(
                {"primitive_id": prim, **item} for item in records
            )

    summary = {
        "schema_version": "glas-cross-meta-primitive-mask-review-v5",
        "cross_meta_eval": str(CROSS_META),
        "cross_meta_eval_sha256": sha256_file(CROSS_META),
        "per_primitive_required": 5,
        "selected_counts": {
            e.primitive_id: len(selected_by_primitive.get(e.primitive_id, []))
            for e in EVALUATIONS
        },
        "all_primitives_complete": all(
            len(selected_by_primitive.get(e.primitive_id, [])) >= 5
            for e in EVALUATIONS
        ),
        "boards": boards,
        "selected_cases": selected_manifest,
        "he_generation_run": False,
        "llm_api_used": False,
    }
    _write_json(args.output_dir / "mask_review_summary.json", summary)
    _write_json(args.output_dir / "attempts_log.json", attempts_log)
    print("DONE", flush=True)
    return 0 if summary["all_primitives_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

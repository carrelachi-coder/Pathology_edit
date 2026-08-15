"""Build/check P1 GLaS/PANDA authority and deterministic-preflight ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.p1_authority_preflight import (
    OUTPUT_FILENAMES,
    build_artifacts,
    validate_committed_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--code-commit",
        help=(
            "Full Git SHA containing this materializer implementation. Required "
            "when writing; --check reuses the committed manifest binding."
        ),
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = REPOSITORY_ROOT
    resources = args.output_dir or root / "phase3_joint_edit_refine" / "resources"
    selection = args.selection or resources / "p1_glas_panda_meta_eval_selection_v1.json"
    source = args.source_manifest or resources / "p1_glas_panda_source_case_pool_v1.json"
    if args.check:
        if args.code_commit or args.selection or args.source_manifest or args.output_dir:
            raise SystemExit(
                "--check validates only the committed default P1 authority artifacts"
            )
        validate_committed_artifacts(root=root, resources=resources)
        return 0
    if not args.code_commit:
        raise SystemExit("--code-commit is required when writing authority artifacts")
    artifacts = build_artifacts(
        root=root,
        selection_path=selection,
        source_manifest_path=source,
        code_commit=args.code_commit,
    )
    resources.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILENAMES.values():
        (resources / filename).write_bytes(artifacts[filename])
    summary = json.loads(artifacts[OUTPUT_FILENAMES["summary"]])
    print(
        json.dumps(
            {
                "authority_manifest": str(resources / OUTPUT_FILENAMES["summary"]),
                "authority_materializer_code_commit": summary[
                    "authority_materializer_code_commit"
                ],
                "frozen_binding_count": summary["frozen_binding_count"],
                "status_counts": summary["status_counts"],
                "visualization_run": summary["visualization_run"],
                "api_used": summary["api_used"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

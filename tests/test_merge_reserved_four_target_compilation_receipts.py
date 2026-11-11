import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.merge_reserved_four_target_compilation_receipts import merge_runs


def write_run(root: Path, name: str, indices: list[int], *, cases_sha: str = "cases") -> Path:
    run = root / name
    run.mkdir()
    (run / "run_identity.json").write_text(
        json.dumps(
            {
                "schema_version": "reserved-four-target-compilation-run-v1",
                "cases": {"sha256": cases_sha},
                "candidate_manifest": {"sha256": "manifest"},
                "selected_case_indices": indices,
            }
        )
    )
    (run / "compilation_receipts.json").write_text(
        json.dumps(
            [
                {
                    "case_index": index,
                    "case_id": f"case-{index}",
                    "terminal_status": "review_required",
                }
                for index in indices
            ]
        )
    )
    return run


def test_merge_orders_disjoint_runs(tmp_path: Path) -> None:
    second = write_run(tmp_path, "second", [1, 2])
    first = write_run(tmp_path, "first", [0])
    receipts, sources = merge_runs([second, first])
    assert [row["case_index"] for row in receipts] == [0, 1, 2]
    assert len(sources) == 2


def test_merge_rejects_duplicate_indices(tmp_path: Path) -> None:
    first = write_run(tmp_path, "first", [0])
    duplicate = write_run(tmp_path, "duplicate", [0])
    with pytest.raises(ValueError, match="duplicate"):
        merge_runs([first, duplicate])


def test_merge_rejects_frozen_input_drift(tmp_path: Path) -> None:
    first = write_run(tmp_path, "first", [0])
    drift = write_run(tmp_path, "drift", [1], cases_sha="changed")
    with pytest.raises(ValueError, match="different frozen"):
        merge_runs([first, drift])


def test_cli_requires_complete_expected_coverage(tmp_path: Path) -> None:
    run = write_run(tmp_path, "run", [0])
    script = Path(__file__).parents[1] / "scripts" / "merge_reserved_four_target_compilation_receipts.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            str(run),
            "--expected-case-count",
            "2",
            "--output",
            str(tmp_path / "merged"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "merged").exists()

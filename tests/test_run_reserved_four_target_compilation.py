import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_reserved_four_target_compilation import validate_inputs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path, *, admitted=False):
    cases = [{"case_id": f"c{i}"} for i in range(8)]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    manifest = {
        "schema_version": "reserved-four-target-candidate-freeze-v1",
        "formal_qualification_updated": admitted,
        "unique_reserved_topology_pass_patients": 2,
        "cases": {"sha256": sha(cases_path)},
        "selected_candidates": [
            {"status": "topology_pass_candidate_only", "case_indices": [0, 1, 2, 3]},
            {"status": "topology_pass_candidate_only", "case_indices": [4, 5, 6, 7]},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return cases_path, manifest_path


def test_validate_inputs_accepts_exact_four_target_mapping(tmp_path: Path):
    cases_path, manifest_path = fixture(tmp_path)
    cases, manifest = validate_inputs(cases_path, manifest_path)
    assert len(cases) == 8
    assert manifest["unique_reserved_topology_pass_patients"] == 2


def test_validate_inputs_rejects_changed_cases(tmp_path: Path):
    cases_path, manifest_path = fixture(tmp_path)
    cases_path.write_text("[]")
    with pytest.raises(ValueError, match="do not match"):
        validate_inputs(cases_path, manifest_path)


def test_validate_inputs_rejects_non_candidate_admission(tmp_path: Path):
    cases_path, manifest_path = fixture(tmp_path, admitted=True)
    with pytest.raises(ValueError, match="improperly claims"):
        validate_inputs(cases_path, manifest_path)


def test_controller_writes_one_terminal_receipt_per_case(tmp_path: Path):
    cases_path, manifest_path = fixture(tmp_path)
    compiler = tmp_path / "scripts" / "fake_compiler.py"
    compiler.parent.mkdir()
    compiler.write_text(
        "import argparse,json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--cases',type=Path);"
        "p.add_argument('--case-index',type=int);p.add_argument('--output',type=Path);"
        "p.add_argument('--device');a=p.parse_args();"
        "c=json.loads(a.cases.read_text())[a.case_index];"
        "(a.output/c['case_id']).mkdir(parents=True);"
        "s={'case_id':c['case_id'],'status':'review_required','reasons':[],"
        "'selected_candidate_id':'candidate'};"
        "(a.output/(c['case_id']+'_summary.json')).write_text(json.dumps(s))\n"
    )
    output = tmp_path / "output"
    controller = Path(__file__).parents[1] / "scripts" / "run_reserved_four_target_compilation.py"
    result = subprocess.run(
        [
            sys.executable,
            str(controller),
            "--cases",
            str(cases_path),
            "--candidate-manifest",
            str(manifest_path),
            "--compiler",
            str(compiler),
            "--output",
            str(output),
            "--device",
            "cpu:0",
            "--device",
            "cpu:1",
            "--seconds-per-case",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipts = json.loads((output / "compilation_receipts.json").read_text())
    assert len(receipts) == 8
    assert {row["terminal_status"] for row in receipts} == {"review_required"}
    assert {row["device"] for row in receipts} == {"cpu:0", "cpu:1"}
    assert len(list((output / "receipts").glob("case_*.json"))) == 8

import json
import subprocess

from phase3_joint_edit_refine.mature_probnet_adapter import (
    _probnet_sampling_audit_rejection,
)


def test_sampling_audit_failure_is_a_variant_rejection(tmp_path) -> None:
    output_path = tmp_path / "nuclei_02.png"
    output_path.write_bytes(b"complete-variant-artifact")
    output_path.with_suffix(".diagnostics.json").write_text(
        json.dumps(
            [
                {
                    "sampling_audit": {
                        "passed": False,
                        "score": 0.7219,
                        "primary_failure_reason": "PROBNET_UNDERFOLLOW",
                        "failure_reasons": ["PROBNET_UNDERFOLLOW"],
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(
        args=["python", "-m", "inpaint_cells.generate"],
        returncode=1,
        stdout="",
        stderr=(
            "RuntimeError: ProbNet count/type/spatial sampling audit failed "
            "after 3 deterministic attempts"
        ),
    )

    rejection = _probnet_sampling_audit_rejection(
        completed=completed,
        output_path=output_path,
    )

    assert rejection is not None
    assert rejection["reasons"] == ["sampling_audit:PROBNET_UNDERFOLLOW"]
    assert rejection["sampling_audit"]["score"] == 0.7219


def test_unrelated_subprocess_error_remains_systemic(tmp_path) -> None:
    output_path = tmp_path / "nuclei_01.png"
    completed = subprocess.CompletedProcess(
        args=["python", "-m", "inpaint_cells.generate"],
        returncode=1,
        stdout="",
        stderr="CUDA out of memory",
    )

    assert (
        _probnet_sampling_audit_rejection(
            completed=completed,
            output_path=output_path,
        )
        is None
    )


def test_audit_error_without_atomic_artifacts_remains_systemic(tmp_path) -> None:
    completed = subprocess.CompletedProcess(
        args=["python", "-m", "inpaint_cells.generate"],
        returncode=1,
        stdout="",
        stderr="ProbNet count/type/spatial sampling audit failed",
    )

    assert (
        _probnet_sampling_audit_rejection(
            completed=completed,
            output_path=tmp_path / "missing.png",
        )
        is None
    )

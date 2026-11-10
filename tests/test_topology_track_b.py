from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from phase3_joint_edit_refine.candidate_feasibility import (
    TissuePortfolioCompilationError,
)
from phase3_joint_edit_refine.tissue_planner import (
    _expected_tissue_authority_binding_sha256,
)
from phase3_joint_edit_refine.topology_track_b import (
    TrackBPortfolioCompilationError,
    compile_tissue_portfolio_with_track_b,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import RefineContractError
from phase3_mask_edit_refine.scene import build_scene_analysis


@dataclass(frozen=True)
class _JointScene:
    tissue: object

    def to_metadata(self):
        return {"tissue": self.tissue.graph.to_metadata()}


class _Preflight:
    def __init__(self, scene, feasible_ids=None):
        self.reference_area_p95 = 9.0
        self._reports = {}
        for interface in scene.tissue.graph.interfaces:
            self._reports[interface.interface_id] = SimpleNamespace(
                interface_id=interface.interface_id,
                cell_feasible_anchor_segment_ids=interface.anchor_segment_ids,
            )
        self.feasible_interface_ids = tuple(
            feasible_ids if feasible_ids is not None else self._reports
        )

    def interface(self, interface_id):
        return self._reports.get(interface_id)

    def to_metadata(self):
        return {
            "reference_area_p95": self.reference_area_p95,
            "feasible_interface_ids": list(self.feasible_interface_ids),
        }


class _Compiler:
    def __init__(self, first_error=None, second_error=None):
        self.first_error = first_error
        self.second_error = second_error
        self.calls = []

    def compile_tissue_portfolio(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1 and self.first_error is not None:
            raise self.first_error
        if len(self.calls) == 2 and self.second_error is not None:
            raise self.second_error
        return SimpleNamespace(name=f"portfolio-{len(self.calls)}")


def _scene_with_two_single_anchor_interfaces():
    tissue = np.full((64, 64), 2, dtype=np.uint8)
    tissue[8:20, 8:20] = 1
    tissue[40:54, 40:54] = 1
    return _JointScene(
        build_scene_analysis(
            tissue,
            schema=MaskProfileSchema.from_reference_profile("BCSS"),
        )
    )


def _bundle():
    front = SimpleNamespace(
        maximum_selected_anchor_fraction=0.75,
        minimum_unselected_anchor_count=1,
    )
    return SimpleNamespace(
        mechanism=SimpleNamespace(
            tissue_program=SimpleNamespace(front=front),
        )
    )


def _error(ids, *, topology_passed=True):
    feedback = {
        "stage": "tissue_area_underfill",
        "topology_passed": topology_passed,
        "remaining_feasible_interface_ids": list(ids),
    }
    return TissuePortfolioCompilationError(
        "ordinary failure",
        vetoes=(),
        structured_failures=(
            {
                "terminal_feedback": feedback,
                "preferred_interface_ids": list(ids),
                "preferred_anchor_ids": [],
                "errors": ["underfill"],
            },
        ),
    )


def _run(monkeypatch, compiler, scene, preflight, fresh_preflight=None):
    import phase3_joint_edit_refine.topology_track_b as module

    bindings = []

    def binding(**kwargs):
        value = {
            "binding_index": len(bindings),
            "preflight_ids": list(kwargs["nuclei_preflight"].feasible_interface_ids),
        }
        bindings.append(value)
        return value

    monkeypatch.setattr(module, "build_tissue_portfolio_authority_binding", binding)
    monkeypatch.setattr(
        module,
        "build_joint_nuclei_preflight",
        lambda **kwargs: fresh_preflight or _Preflight(kwargs["scene"]),
    )
    result = compile_tissue_portfolio_with_track_b(
        compiler=compiler,
        joint_case=object(),
        tissue_case=object(),
        source_tissue=np.zeros((64, 64), dtype=np.uint8),
        schema=object(),
        scene=scene,
        tissue_bundle=object(),
        joint_bundle=_bundle(),
        allocation=object(),
        nuclei_preflight=preflight,
    )
    return result, bindings


def test_ordinary_success_never_invokes_track_b(monkeypatch):
    scene = _scene_with_two_single_anchor_interfaces()
    compiler = _Compiler()
    result, bindings = _run(monkeypatch, compiler, scene, _Preflight(scene))
    assert result.track_b_used is False
    assert result.scene is scene
    assert len(compiler.calls) == 1
    assert len(bindings) == 1


def test_topology_failure_never_invokes_track_b(monkeypatch):
    scene = _scene_with_two_single_anchor_interfaces()
    ids = _Preflight(scene).feasible_interface_ids
    original = _error(ids, topology_passed=False)
    compiler = _Compiler(original)
    with pytest.raises(TissuePortfolioCompilationError) as caught:
        _run(monkeypatch, compiler, scene, _Preflight(scene))
    assert caught.value is original
    assert len(compiler.calls) == 1


def test_no_eligible_interface_preserves_original_failure(monkeypatch):
    scene = _scene_with_two_single_anchor_interfaces()
    preflight = _Preflight(scene)
    interface_id = preflight.feasible_interface_ids[0]
    preflight._reports[interface_id] = SimpleNamespace(
        interface_id=interface_id,
        cell_feasible_anchor_segment_ids=(),
    )
    original = _error((interface_id,))
    compiler = _Compiler(original)
    with pytest.raises(TissuePortfolioCompilationError) as caught:
        _run(monkeypatch, compiler, scene, preflight)
    assert caught.value is original
    assert len(compiler.calls) == 1


def test_multiple_eligible_interfaces_are_rebuilt_once_with_fresh_identity(monkeypatch):
    scene = _scene_with_two_single_anchor_interfaces()
    preflight = _Preflight(scene)
    ids = preflight.feasible_interface_ids
    compiler = _Compiler(_error(tuple(reversed(ids))))
    result, bindings = _run(monkeypatch, compiler, scene, preflight)

    assert result.track_b_used is True
    assert len(compiler.calls) == 2
    assert len(bindings) == 2
    assert bindings[0] != bindings[1]
    assert compiler.calls[0]["scene"] is scene
    assert compiler.calls[1]["scene"] is result.scene
    assert compiler.calls[1]["nuclei_preflight"] is result.nuclei_preflight
    assert compiler.calls[1]["authority_binding"]["binding_index"] == bindings[1][
        "binding_index"
    ]
    transformation_authority = compiler.calls[1]["authority_binding"][
        "track_b_transformation_authority"
    ]
    assert transformation_authority["original_scene_sha256"] != (
        transformation_authority["transformed_scene_sha256"]
    )
    assert result.authority_binding == compiler.calls[1]["authority_binding"]
    records = result.audit["transformed_interfaces"]
    assert [item["interface_id"] for item in records] == sorted(ids)
    assert result.audit["rounds"] == 1
    assert result.audit["recursive_subdivision"] is False
    for record in records:
        assert len(record["new_subanchors"]) == 2
        assert all(
            item["contact_pixels"] >= record["equivalent_diameter_pixels"]
            for item in record["new_subanchors"]
        )


def test_fresh_preflight_rejection_does_not_reuse_parent_feasibility(monkeypatch):
    scene = _scene_with_two_single_anchor_interfaces()
    preflight = _Preflight(scene)
    interface_id = preflight.feasible_interface_ids[0]
    original = _error((interface_id,))
    compiler = _Compiler(original)
    fresh = _Preflight(scene, feasible_ids=())
    with pytest.raises(TissuePortfolioCompilationError) as caught:
        _run(monkeypatch, compiler, scene, preflight, fresh_preflight=fresh)
    assert caught.value is original
    assert len(compiler.calls) == 1


def test_fresh_track_b_failure_retains_both_audited_failures(monkeypatch):
    scene = _scene_with_two_single_anchor_interfaces()
    preflight = _Preflight(scene)
    interface_id = preflight.feasible_interface_ids[0]
    original = _error((interface_id,))
    transformed = _error((interface_id,))
    compiler = _Compiler(original, transformed)
    with pytest.raises(TrackBPortfolioCompilationError) as caught:
        _run(monkeypatch, compiler, scene, preflight)
    assert caught.value.original_error is original
    assert caught.value.track_b_error is transformed
    assert caught.value.audit["rounds"] == 1
    assert caught.value.audit["track_b_failure"]["message"] == str(transformed)


def test_planner_preserves_valid_track_b_authority_extension(monkeypatch):
    import phase3_joint_edit_refine.tissue_planner as module

    base = {"schema_version": "test", "nuclei_preflight_sha256": "preflight"}
    extension = {
        "workflow_version": "workflow-v1",
        "candidate_version": "candidate-v1",
        "original_scene_sha256": "a" * 64,
        "transformed_scene_sha256": "b" * 64,
        "transformed_interfaces_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        module,
        "build_tissue_portfolio_authority_binding",
        lambda **kwargs: dict(base),
    )
    monkeypatch.setattr(
        module,
        "load_id_mask",
        lambda path: np.zeros((2, 2), dtype=np.uint8),
    )
    portfolio = SimpleNamespace(
        authority_binding={
            **base,
            "track_b_transformation_authority": extension,
        }
    )
    digest = _expected_tissue_authority_binding_sha256(
        case=SimpleNamespace(source_mask_uri="unused"),
        joint_case=object(),
        joint_bundle=object(),
        tissue_bundle=object(),
        allocation=object(),
        nuclei_preflight=object(),
        candidate_portfolio=portfolio,
    )
    assert digest == module.canonical_metadata_sha256(portfolio.authority_binding)


@pytest.mark.parametrize(
    "extension",
    [
        {"workflow_version": "workflow-v1"},
        {
            "workflow_version": "workflow-v1",
            "candidate_version": "candidate-v1",
            "original_scene_sha256": "a" * 64,
            "transformed_scene_sha256": "a" * 64,
            "transformed_interfaces_sha256": "c" * 64,
        },
    ],
)
def test_planner_rejects_invalid_track_b_authority_extension(
    monkeypatch, extension
):
    import phase3_joint_edit_refine.tissue_planner as module

    base = {"schema_version": "test"}
    monkeypatch.setattr(
        module,
        "build_tissue_portfolio_authority_binding",
        lambda **kwargs: dict(base),
    )
    monkeypatch.setattr(
        module,
        "load_id_mask",
        lambda path: np.zeros((2, 2), dtype=np.uint8),
    )
    portfolio = SimpleNamespace(
        authority_binding={
            **base,
            "track_b_transformation_authority": extension,
        }
    )
    with pytest.raises(RefineContractError, match="transformation authority"):
        _expected_tissue_authority_binding_sha256(
            case=SimpleNamespace(source_mask_uri="unused"),
            joint_case=object(),
            joint_bundle=object(),
            tissue_bundle=object(),
            allocation=object(),
            nuclei_preflight=object(),
            candidate_portfolio=portfolio,
        )

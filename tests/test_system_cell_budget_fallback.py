"""System-selected cell counts may yield to mask capacity, explicit counts may not."""

from types import SimpleNamespace

import numpy as np

from phase3_joint_edit_refine.gates import _cell_quota, _local_population_density
from phase3_joint_edit_refine.models import CellCountExtentBudget
from phase3_joint_edit_refine.workflow import _apply_profile_visible_cell_budget


def _context(*, derived: bool):
    primitive_id = "cell-type-abundance-increase-v1"
    trace = {
        "biological_desired_count": 16,
        "desired_count": 16,
        "resolved_count": 10,
        "requested_count": 10,
        "placed_count": 10,
        "batch_max_attainable_count": 10,
        "capacity_max_count": 10,
        "cell_capacity_certified": True,
        "cell_capacity_fallback_used": True,
    }
    case = SimpleNamespace(
        primitive_id=primitive_id,
        pathology_domain_id="melanoma-v1",
        semantic_intent={
            "derived_budget_policies": {primitive_id: {"authority": "system_owned_profile_specific_budget"}}
        } if derived else {},
        cell_count_extent_budget=CellCountExtentBudget(16, 12, 24, 384),
    )
    primitive = SimpleNamespace(minimum_effect_delta_count_for=lambda domain: 8, scope="cell_only")
    return SimpleNamespace(
        case=case,
        bundle=SimpleNamespace(primitive=primitive),
        candidate=SimpleNamespace(
            tool_trace=trace,
            generation_support=np.ones((16, 16), dtype=bool),
            target_nuclei_mask=np.zeros((16, 16), dtype=np.uint8),
            ledger=SimpleNamespace(added_instance_ids=tuple(range(10))),
        ),
        source_nuclei=np.zeros((16, 16), dtype=np.uint8),
        plan=SimpleNamespace(cell_plan=SimpleNamespace(mechanism_quota_role="explicit_increment", baseline_mode="structured_add")),
        executable_contract=SimpleNamespace(packing_certificate={"passed": True, "minimum_safe_count": 12}),
    )


def test_system_count_accepts_ten_valid_cells_with_disclosed_capacity_limit():
    context = _context(derived=True)
    quota = _cell_quota(context)
    density = _local_population_density(context)
    assert quota.passed
    assert density.passed
    assert quota.metrics["system_budget_capacity_limited"]
    assert density.metrics["effective_min_delta"] == 10
    assert quota.metrics["placed_count"] == 10


def test_explicit_twelve_cell_minimum_does_not_silently_become_ten():
    context = _context(derived=False)
    assert not _cell_quota(context).passed
    assert not _local_population_density(context).passed


def test_profile_visible_target_is_capped_by_observed_zone_capacity():
    case = SimpleNamespace(annotation_profile_id="puma-semantic-v1")
    budget, metadata = _apply_profile_visible_cell_budget(
        case,
        primitive_id="cell-type-abundance-increase-v1",
        minimum_delta_count=8,
        budget=CellCountExtentBudget(6, 4, 8, 96),
        metadata={"selected_zone_executable_capacity": 10},
    )
    assert (budget.target_delta_count, budget.min_delta_count, budget.max_delta_count) == (10, 8, 10)
    assert metadata["capacity_limited_visible_target"] == 10

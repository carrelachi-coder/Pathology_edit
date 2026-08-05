"""Fail-closed deterministic gates for atomic tissue--cell candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import GateReport

from .models import (
    JointCandidate,
    JointCaseContext,
    JointEditPlan,
    JointGateCheck,
    JointGateReport,
)
from .nuclei import iter_instances
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle


BASE_REQUIRED_CHECKS = (
    "primitive_semantics",
    "tool_program_binding",
    "tissue_gate_binding",
    "whole_instance_changes",
    "protected_nuclei_preserved",
    "nuclei_overlap",
    "nuclei_tissue_containment",
    "reference_shape_integrity",
    "cell_quota",
    "cell_spatial_distribution",
    "joint_area",
    "tissue_floor",
    "cell_tissue_compatibility",
    "cell_zone_localization",
    "joint_provenance",
)


@dataclass(frozen=True)
class JointGateContext:
    case: JointCaseContext
    source_tissue: np.ndarray
    source_nuclei: np.ndarray
    schema: MaskProfileSchema
    scene: JointSceneAnalysis
    bundle: JointSkillBundle
    plan: JointEditPlan
    candidate: JointCandidate
    tissue_gate_report: GateReport


class JointGateRegistry:
    def __init__(self) -> None:
        self._checks: dict[str, Callable[[JointGateContext], JointGateCheck]] = {
            "primitive_semantics": _primitive_semantics,
            "tool_program_binding": _tool_program_binding,
            "tissue_gate_binding": _tissue_gate_binding,
            "native_structure_preserved": _native_structure_preserved,
            "whole_instance_changes": _whole_instance_changes,
            "protected_nuclei_preserved": _protected_nuclei_preserved,
            "nuclei_overlap": _nuclei_overlap,
            "nuclei_tissue_containment": _nuclei_tissue_containment,
            "reference_shape_integrity": _reference_shape_integrity,
            "cell_quota": _cell_quota,
            "cell_spatial_distribution": _cell_spatial_distribution,
            "joint_area": _joint_area,
            "tissue_floor": _tissue_floor,
            "cell_tissue_compatibility": _cell_tissue_compatibility,
            "cell_zone_localization": _cell_zone_localization,
            "joint_provenance": _joint_provenance,
            "profile_provenance": _profile_provenance,
            "prohibited_cell_region": _prohibited_cell_region,
            "prohibited_generation_support": _prohibited_generation_support,
            "orca_fragment_protection": _orca_fragment_protection,
            "fine_pattern_preserved": _fine_pattern_preserved,
            "local_shape_distribution": _local_shape_distribution,
            "local_population_density": _local_population_density,
            "interface_seam_continuity": _interface_seam_continuity,
            "mechanism_realization": _mechanism_realization,
        }
        missing = sorted(set(BASE_REQUIRED_CHECKS) - set(self._checks))
        if missing:
            raise RuntimeError("required joint gate implementations missing: " + ", ".join(missing))

    @property
    def available_checker_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._checks))

    def run(self, context: JointGateContext) -> JointGateReport:
        requested = list(BASE_REQUIRED_CHECKS)
        for check_id in context.bundle.required_checker_ids:
            if check_id not in requested:
                requested.append(check_id)
        missing = [check_id for check_id in requested if check_id not in self._checks]
        if missing:
            checks = tuple(_result(item, False, "required checker is not registered") for item in missing)
            return JointGateReport(context.candidate.candidate_id, False, checks)
        checks = tuple(self._checks[item](context) for item in requested)
        return JointGateReport(
            context.candidate.candidate_id,
            all(item.passed for item in checks if item.severity == "hard"),
            checks,
        )


def _result(check_id, passed, detail, *, metrics=None, severity="hard"):
    return JointGateCheck(check_id, bool(passed), severity, detail, metrics or {})


def _primitive_semantics(c):
    primitive = c.bundle.primitive
    tissue_pixels = c.candidate.ledger.tissue_pixels
    cell_pixels = c.candidate.ledger.cell_pixels
    baseline_ok = c.plan.cell_plan.baseline_mode in primitive.allowed_baseline_modes
    quota_ok = c.plan.cell_plan.mechanism_quota_role in primitive.allowed_quota_roles
    if primitive.scope == "cell_only":
        scope_ok = tissue_pixels == 0 and cell_pixels > 0 and c.plan.tissue_plan is None
        budget_ok = c.case.cell_count_extent_budget is not None
    else:
        scope_ok = tissue_pixels > 0 and c.plan.tissue_plan is not None
        budget_ok = True
    passed = scope_ok and baseline_ok and quota_ok and budget_ok
    return _result(
        "primitive_semantics",
        passed,
        "primitive scope, baseline, quota and budget are bound"
        if passed
        else "candidate does not realize the requested primitive contract",
        metrics={
            "primitive_scope": primitive.scope,
            "tissue_pixels": tissue_pixels,
            "cell_pixels": cell_pixels,
            "baseline_mode": c.plan.cell_plan.baseline_mode,
            "quota_role": c.plan.cell_plan.mechanism_quota_role,
            "scope_ok": scope_ok,
            "baseline_ok": baseline_ok,
            "quota_ok": quota_ok,
            "budget_ok": budget_ok,
        },
    )


def _tool_program_binding(c):
    program = c.candidate.tool_trace.get("compiled_cell_tool_program")
    expected = c.plan.cell_plan
    passed = bool(
        isinstance(program, dict)
        and program.get("compiler_version") == "joint-cell-tool-compiler-v2"
        and program.get("primitive_id") == c.case.primitive_id
        and program.get("mechanism_id") == c.plan.selected_mechanism_id
        and program.get("program_id") == expected.tool_program_id
        and program.get("baseline_mode") == expected.baseline_mode
        and program.get("mechanism_program_id") == expected.mechanism_program_id
        and program.get("quota_role") == expected.mechanism_quota_role
        and all(
            int(program.get(f"{name}_pixels", 0)) >= 0
            for name in (
                "erasure_region",
                "placement_center_region",
                "valid_footprint_region",
                "support_context_region",
                "mechanism_region",
            )
        )
    )
    return _result(
        "tool_program_binding",
        passed,
        "candidate binds the Planner plan to a compiled E/P/V/S program"
        if passed
        else "compiled E/P/V/S program is missing or inconsistent",
        metrics={"compiled_program": program or {}},
    )


def _tissue_gate_binding(c):
    passed = c.tissue_gate_report.passed and c.tissue_gate_report.candidate_id == c.candidate.tissue_candidate_id
    return _result("tissue_gate_binding", passed, "paired tissue candidate passed its complete deterministic gate report" if passed else "paired tissue gate failed or candidate ID does not bind")


def _native_structure_preserved(c):
    required = set(c.bundle.mechanism.representability.required_auxiliary_structures)
    available = set(c.scene.auxiliary_structure_masks)
    missing = sorted(required - available)
    tissue_ok = c.tissue_gate_report.passed
    violations = {
        structure_id: int(
            np.count_nonzero(
                c.candidate.generation_support
                & c.scene.auxiliary_structure_masks[structure_id]
            )
        )
        for structure_id in sorted(required - set(missing))
    }
    passed = tissue_ok and not missing and not any(violations.values())
    return _result(
        "native_structure_preserved",
        passed,
        (
            "required auxiliary structures are present and outside joint generation support"
            if passed
            else f"missing auxiliary structures={missing}, overlap={violations}, or tissue gate failed"
        ),
        metrics={
            "required": sorted(required),
            "missing": missing,
            "generation_support_overlap_pixels": violations,
        },
    )


def _whole_instance_changes(c):
    partial = c.candidate.tool_trace.get("partial_source_instance_ids", [])
    passed = c.candidate.tool_trace.get("whole_instance_changes") is True and not partial
    return _result("whole_instance_changes", passed, "all source nucleus edits remove complete instances" if passed else f"partial source instances={partial}", metrics={"partial_source_instance_ids": partial})


def _protected_nuclei_preserved(c):
    target = c.candidate.target_nuclei_mask
    violations = 0
    missing_ids = []
    for instance_id in c.plan.cell_plan.protected_instance_ids:
        component = c.scene.instance_masks.get(instance_id)
        if component is None:
            missing_ids.append(instance_id)
            continue
        violations += int(np.count_nonzero(target[component] != c.source_nuclei[component]))
    exterior = ~c.candidate.generation_support
    exterior_violations = int(np.count_nonzero(target[exterior] != c.source_nuclei[exterior]))
    passed = not missing_ids and violations == 0 and exterior_violations == 0
    return _result("protected_nuclei_preserved", passed, "protected and generation-support-exterior nuclei are pixel-exact" if passed else "protected/exterior nuclei changed", metrics={"missing_instance_ids": missing_ids, "protected_violation_pixels": violations, "exterior_violation_pixels": exterior_violations})


def _nuclei_overlap(c):
    overlap = int(c.candidate.tool_trace.get("overlap_pixels", -1))
    passed = overlap == 0
    return _result("nuclei_overlap", passed, "layout tool certified zero overlap" if passed else "layout overlap trace missing or nonzero", metrics={"overlap_pixels": overlap})


def _nuclei_tissue_containment(c):
    source_nuclei = c.source_nuclei > 0
    nuclei = c.candidate.target_nuclei_mask > 0
    prohibited = np.isin(c.candidate.target_tissue_mask, c.bundle.annotation_profile.prohibit_cell_placement_fine_ids)
    # A source observation can already contain CellViT pixels over a profile's
    # non-tissue/ignore label.  The editor must preserve that baseline outside
    # its support, but it must not claim the pre-existing disagreement as a new
    # placement error.  Only newly occupied pixels (or retained pixels made
    # newly illegal by the tissue edit) are attributable to this candidate.
    source_prohibited = np.isin(
        c.source_tissue,
        c.bundle.annotation_profile.prohibit_cell_placement_fine_ids,
    )
    newly_occupied = nuclei & ~source_nuclei
    newly_illegal_retained = nuclei & source_nuclei & prohibited & ~source_prohibited
    violations = int(
        np.count_nonzero((newly_occupied & prohibited) | newly_illegal_retained)
    )
    border_instances = sum(1 for _, _, component in iter_instances(c.candidate.target_nuclei_mask) if _touches_border(component))
    source_border_ids = {item.instance_id for item in c.scene.cells.instances if item.touches_border}
    # Existing protected border nuclei are allowed; no newly placed instance may be clipped.
    new_border = max(0, border_instances - len(source_border_ids))
    passed = violations == 0 and new_border == 0
    return _result(
        "nuclei_tissue_containment",
        passed,
        "all newly added or newly exposed nuclei are contained in legal tissue"
        if passed
        else "a new/newly exposed nucleus overlaps prohibited tissue or a new nucleus is clipped",
        metrics={
            "prohibited_pixels": violations,
            "baseline_prohibited_nucleus_pixels": int(
                np.count_nonzero(source_nuclei & source_prohibited)
            ),
            "new_border_instances": new_border,
        },
    )


def _reference_shape_integrity(c):
    eligible = set(c.candidate.tool_trace.get("reference_shape_ids", ()))
    rejected = c.candidate.tool_trace.get("reference_shape_rejections", {})
    placements = c.candidate.tool_trace.get("placements", ())
    selected = {
        item.get("reference_instance_id")
        for item in placements
        if isinstance(item, dict)
    }
    missing_bindings = sorted(
        str(value) for value in selected if not value or value not in eligible
    )
    rejected_selected = sorted(
        str(value)
        for value in selected
        if isinstance(rejected, dict) and value in rejected
    )
    border_rejections = sorted(
        instance_id
        for instance_id, reason in rejected.items()
        if reason == "patch_boundary_censored_shape"
    ) if isinstance(rejected, dict) else []
    shape_sampling = c.candidate.tool_trace.get("shape_sampling", {})
    mature = c.candidate.tool_trace.get("mature_probnet_contract") is True
    size_calibration = (
        shape_sampling.get("library_size_calibration", {})
        if isinstance(shape_sampling, dict)
        else {}
    )
    uncalibrated = (
        size_calibration.get("uncalibrated_no_reference_by_type", {})
        if isinstance(size_calibration, dict)
        else {}
    )
    uncalibrated_count = (
        sum(int(value) for value in uncalibrated.values())
        if isinstance(uncalibrated, dict)
        else -1
    )
    mature_policy_ok = bool(
        not mature
        or (
            isinstance(shape_sampling, dict)
            and shape_sampling.get("policy")
            in {
                "same_class_reference_without_replacement_then_library",
                "component_local_same_class_reference_then_component_calibrated_library",
            }
            and uncalibrated_count == 0
        )
    )
    passed = bool(
        c.candidate.tool_trace.get("reference_shape_integrity_certified") is True
        and eligible
        and not missing_bindings
        and not rejected_selected
        and mature_policy_ok
    )
    return _result(
        "reference_shape_integrity",
        passed,
        (
            "all added nuclei bind to complete non-border source instances"
            if passed
            else "reference template provenance is missing or includes a censored source shape"
        ),
        metrics={
            "eligible_reference_count": len(eligible),
            "selected_reference_ids": sorted(value for value in selected if value),
            "border_censored_source_ids": border_rejections,
            "missing_reference_bindings": missing_bindings,
            "selected_rejected_reference_ids": rejected_selected,
            "mature_shape_policy_ok": mature_policy_ok,
            "uncalibrated_library_fallback_count": uncalibrated_count,
        },
    )


def _cell_quota(c):
    desired = int(c.candidate.tool_trace.get("desired_count", -1))
    resolved = int(c.candidate.tool_trace.get("resolved_count", -1))
    requested = int(c.candidate.tool_trace.get("requested_count", -1))
    placed = int(c.candidate.tool_trace.get("placed_count", -1))
    batch_max = int(c.candidate.tool_trace.get("batch_max_attainable_count", -1))
    certified = c.candidate.tool_trace.get("cell_capacity_certified") is True
    fallback = bool(c.candidate.tool_trace.get("cell_capacity_fallback_used"))
    completion = placed / max(1, resolved)
    exact = (
        desired >= 0
        and resolved >= 0
        and requested == resolved
        and placed == resolved
        and (resolved == desired or fallback)
    )
    maximum_safe_fallback = (
        fallback
        and certified
        and 0 <= resolved < desired
        and resolved == batch_max
    )
    passed = exact and (resolved == desired or maximum_safe_fallback)
    return _result(
        "cell_quota",
        passed,
        (
            "cell placement completed the desired quota"
            if passed and resolved == desired
            else (
                "cell placement completed the deterministically proven maximum reachable quota"
                if passed
                else "cell layout did not bind its desired, resolved and placed quotas"
            )
        ),
        metrics={
            "desired_count": desired,
            "resolved_count": resolved,
            "requested_count": requested,
            "placed_count": placed,
            "batch_max_attainable_count": batch_max,
            "completion": completion,
            "used_capacity_fallback": maximum_safe_fallback,
            "capacity_certified": certified,
        },
    )


def _cell_spatial_distribution(c):
    source = _spatial_summary(c.source_nuclei)
    target = _spatial_summary(c.candidate.target_nuclei_mask)
    nnd_ratio = _safe_ratio(target["mean_nnd_px"], source["mean_nnd_px"])
    count_ratio = _safe_ratio(target["instance_count"], source["instance_count"])
    area_ratio = _safe_ratio(target["instance_area_p95"], source["instance_area_p95"])
    ripley_ratio = _safe_ratio(target["ripley_k24_normalized"], source["ripley_k24_normalized"])
    finite = all(
        value is None or np.isfinite(value)
        for value in (nnd_ratio, count_ratio, area_ratio, ripley_ratio)
    )
    # Research fallback only. Production is additionally bound to the mature
    # patch-adaptive sampling audit by ``local_population_density``.
    checks = {
        "nnd_ratio": nnd_ratio is None or 0.25 <= nnd_ratio <= 4.0,
        "instance_count_ratio": count_ratio is not None and 0.15 <= count_ratio <= 6.0,
        "instance_area_p95_ratio": area_ratio is None or 0.10 <= area_ratio <= 10.0,
        "ripley_k24_ratio": ripley_ratio is None or 0.05 <= ripley_ratio <= 20.0,
    }
    passed = finite and bool(target["instance_count"]) and all(checks.values())
    return _result(
        "cell_spatial_distribution",
        passed,
        (
            "density, NND, component-area and Ripley-K summaries pass the research safety envelope"
            if passed
            else "degenerate or extreme joint cell spatial distribution"
        ),
        metrics={
            "calibration_status": "uncalibrated_research_envelope",
            "source": source,
            "target": target,
            "ratios": {
                "mean_nnd": nnd_ratio,
                "instance_count": count_ratio,
                "instance_area_p95": area_ratio,
                "ripley_k24_normalized": ripley_ratio,
            },
            "range_checks": checks,
        },
    )


def _local_shape_distribution(c):
    placements = c.candidate.tool_trace.get("placements", ())
    added_areas = [
        float(item["area_px"])
        for item in placements
        if isinstance(item, dict) and float(item.get("area_px", 0)) > 0
    ]
    reference_ids = set(c.candidate.tool_trace.get("reference_shape_ids", ()))
    local_areas = [
        float(item.area_px)
        for item in c.scene.cells.instances
        if item.instance_id in reference_ids
        and not item.touches_border
        and "merged_suspect" not in item.quality_flags
    ]
    mature = c.candidate.tool_trace.get("mature_probnet_contract") is True
    shape_sampling = c.candidate.tool_trace.get("shape_sampling", {})
    calibration = (
        shape_sampling.get("library_size_calibration", {})
        if isinstance(shape_sampling, dict)
        else {}
    )
    uncalibrated = (
        calibration.get("uncalibrated_no_reference_by_type", {})
        if isinstance(calibration, dict)
        else {}
    )
    mature_certified = bool(
        mature
        and isinstance(shape_sampling, dict)
        and c.candidate.tool_trace.get("reference_shape_integrity_certified") is True
        and isinstance(uncalibrated, dict)
        and sum(int(value) for value in uncalibrated.values()) == 0
    )
    if not added_areas and int(c.candidate.tool_trace.get("placed_count", 0)) == 0:
        passed = bool(local_areas)
        ratio = None
    elif not added_areas and mature_certified:
        passed = True
        ratio = None
    else:
        ratio = _safe_ratio(
            float(np.median(added_areas)) if added_areas else None,
            float(np.median(local_areas)) if local_areas else None,
        )
        passed = ratio is not None and 0.60 <= ratio <= 1.67
    return _result(
        "local_shape_distribution",
        passed,
        "added shapes match complete same-class source shapes"
        if passed
        else "added nucleus size is unsupported by local complete references",
        metrics={
            "added_count": len(added_areas),
            "local_reference_count": len(local_areas),
            "median_area_ratio": ratio,
            "mature_shape_sampling_certified": mature_certified,
        },
    )


def _local_population_density(c):
    audit = c.candidate.tool_trace.get("sampling_audit", {})
    if c.candidate.tool_trace.get("mature_probnet_contract") is True:
        passed = isinstance(audit, dict) and audit.get("passed") is True
        return _result(
            "local_population_density",
            passed,
            "mature patch-adaptive count/type/spatial audit passed"
            if passed
            else "mature ProbNet sampling audit is absent or failed",
            metrics={"mature_sampling_audit": audit},
        )
    region = np.asarray(c.candidate.generation_support, dtype=bool)
    source_count = _instance_centers_in_region(c.source_nuclei, region)
    target_count = _instance_centers_in_region(
        c.candidate.target_nuclei_mask, region
    )
    if c.bundle.primitive.scope == "cell_only":
        budget = c.case.cell_count_extent_budget
        delta = target_count - source_count
        passed = bool(
            budget
            and budget.min_delta_count <= delta <= budget.max_delta_count
        )
        metrics = {
            "source_count": source_count,
            "target_count": target_count,
            "delta_count": delta,
            "allowed_delta": (
                [budget.min_delta_count, budget.max_delta_count] if budget else None
            ),
        }
    else:
        source_density = source_count / max(1, int(np.count_nonzero(region)))
        target_density = target_count / max(1, int(np.count_nonzero(region)))
        ratio = _safe_ratio(target_density, source_density)
        # A no-source region is judged by exact quota and local reference
        # availability; otherwise keep a tight but non-pathology-specific band.
        passed = (
            target_count == int(c.candidate.tool_trace.get("placed_count", -1))
            if source_count == 0
            else ratio is not None and 0.50 <= ratio <= 2.0
        )
        metrics = {
            "source_count": source_count,
            "target_count": target_count,
            "density_ratio": ratio,
            "calibration": "patch_local_research_fallback",
        }
    return _result(
        "local_population_density",
        passed,
        "local population contract passed"
        if passed
        else "local density/count delta is inconsistent with the primitive",
        metrics=metrics,
    )


def _interface_seam_continuity(c):
    if c.bundle.primitive.scope == "cell_only":
        return _result(
            "interface_seam_continuity",
            True,
            "cell-only primitive has no artificial tissue seam",
            metrics={"applicable": False},
        )
    change = np.asarray(c.candidate.tissue_change, dtype=bool)
    seam = change & ndimage.binary_dilation(~change, iterations=1)
    diameter = float(c.scene.population.nominal_nucleus_diameter_px or 8.0)
    band = ndimage.binary_dilation(seam, iterations=max(1, int(round(diameter))))
    inner = change & band
    if c.plan.tissue_plan is not None:
        target_ids = c.schema.resolve_fine_ids(c.plan.tissue_plan.target_label)
        outer = (
            ~change
            & band
            & np.isin(c.candidate.target_tissue_mask, target_ids)
        )
    else:
        outer = ~change & band
    target = c.candidate.target_nuclei_mask
    inner_density = _instance_centers_in_region(target, inner) / max(
        1, int(np.count_nonzero(inner))
    )
    outer_density = _instance_centers_in_region(target, outer) / max(
        1, int(np.count_nonzero(outer))
    )
    ratio = (
        _safe_ratio(inner_density, outer_density)
        if np.count_nonzero(outer)
        else None
    )
    audit = c.candidate.tool_trace.get("sampling_audit", {})
    mature_pass = bool(
        c.candidate.tool_trace.get("mature_probnet_contract") is True
        and isinstance(audit, dict)
        and audit.get("passed") is True
    )
    passed = mature_pass or ratio is None or 0.25 <= ratio <= 4.0
    return _result(
        "interface_seam_continuity",
        passed,
        "new and retained populations do not collapse at the edit seam"
        if passed
        else "cell density has an artificial discontinuity at the edit seam",
        metrics={
            "applicable": True,
            "inner_density": inner_density,
            "outer_density": outer_density,
            "inner_outer_ratio": ratio,
            "mature_sampling_audit_passed": mature_pass,
        },
    )


def _mechanism_realization(c):
    placements = [
        item
        for item in c.candidate.tool_trace.get("placements", ())
        if isinstance(item, dict)
    ]
    layout = c.plan.cell_plan.mechanism_program_id
    allowed = c.bundle.mechanism.cell_program.layout_programs
    cluster_min, cluster_max = c.bundle.mechanism.cell_program.cluster_size_range
    declared_sizes = [
        int(item.get("cluster_size", 1)) for item in placements
    ]
    sizes_ok = all(cluster_min <= value <= cluster_max for value in declared_sizes)
    mature_baseline_only = (
        c.candidate.tool_trace.get("mature_probnet_contract") is True
        and layout == "population_replacement"
    )
    modifier_certified = (
        c.candidate.tool_trace.get("mechanism_modifier_certified") is True
    )
    passed = layout in allowed and sizes_ok and (
        bool(placements) or mature_baseline_only or modifier_certified
    )
    return _result(
        "mechanism_realization",
        passed,
        "layout family and cluster cardinality realize the selected mechanism"
        if passed
        else "cell placements do not prove the selected mechanism program",
        metrics={
            "layout_program": layout,
            "allowed_layouts": list(allowed),
            "placement_count": len(placements),
            "cluster_size_range": [cluster_min, cluster_max],
            "declared_cluster_sizes": declared_sizes,
            "mature_baseline_only": mature_baseline_only,
            "mechanism_modifier_certified": modifier_certified,
        },
    )


def _joint_area(c):
    if c.bundle.primitive.budget_mode == "count_extent":
        budget = c.case.cell_count_extent_budget
        extent = _maximum_changed_distance_to_interfaces(c)
        passed = bool(budget and extent <= budget.maximum_extent_px)
        return _result(
            "joint_area",
            passed,
            "cell-only edit satisfies its count/extent budget"
            if passed
            else "cell-only change exceeds its declared extent budget",
            metrics={
                "budget_mode": "count_extent",
                "maximum_observed_extent_px": extent,
                "maximum_allowed_extent_px": (
                    budget.maximum_extent_px if budget else None
                ),
            },
        )
    budget = c.case.joint_area_budget
    hard_min, hard_max = budget.hard_interval_pixels(c.candidate.joint_change.shape)
    desired_min, desired_max = budget.desired_interval_pixels(c.candidate.joint_change.shape)
    actual = c.candidate.ledger.joint_pixels
    in_desired = desired_min <= actual <= desired_max
    tissue_trace = c.candidate.tool_trace.get("tissue_tool_trace", {})
    tissue_fallback = (
        isinstance(tissue_trace, dict)
        and int(tissue_trace.get("resolved_target_pixels", actual))
        < int(tissue_trace.get("desired_target_pixels", actual))
    )
    capacity_exhausted = bool(c.candidate.tool_trace.get("placement_capacity_exhausted")) or tissue_fallback
    batch_max = int(c.candidate.tool_trace.get("batch_max_safe_joint_pixels", -1))
    fallback = (
        budget.fallback_policy == "max_feasible_below_target"
        and hard_min <= actual < desired_min
        and capacity_exhausted
        and actual == batch_max
    )
    adaptive_tissue = _proven_max_safe_tissue_fallback(c)
    adaptive_fallback = (
        adaptive_tissue
        and 0 < actual <= hard_max
        and actual < desired_min
        and actual == batch_max
    )
    passed = in_desired or fallback or adaptive_fallback
    detail = (
        "joint union is in target tolerance"
        if in_desired
        else (
            "joint union accepted as explicit maximum-safe fallback"
            if fallback
            else (
                "joint union accepted below the standard floor from a proven maximum-safe tissue solve"
                if adaptive_fallback
                else "joint union violates target or was not proven maximum-safe"
            )
        )
    )
    return _result(
        "joint_area",
        passed,
        detail,
        metrics={
            "actual_pixels": actual,
            "hard_interval": [hard_min, hard_max],
            "desired_interval": [desired_min, desired_max],
            "used_fallback": fallback or adaptive_fallback,
            "used_capacity_adaptive_fallback": adaptive_fallback,
            "capacity_exhausted": capacity_exhausted,
            "batch_max_safe_joint_pixels": batch_max,
        },
    )


def _tissue_floor(c):
    actual = c.candidate.ledger.tissue_pixels
    applies = (
        c.bundle.primitive.scope == "tissue_and_cell"
        and c.bundle.mechanism.coupling.tissue_floor_applies
    )
    floor = (
        c.case.joint_area_budget.tissue_floor_pixels(
            c.candidate.tissue_change.shape
        )
        if applies and c.case.joint_area_budget is not None
        else 0
    )
    adaptive_fallback = applies and actual < floor and _proven_max_safe_tissue_fallback(c)
    passed = not applies or actual >= floor or adaptive_fallback
    return _result(
        "tissue_floor",
        passed,
        (
            "burden edit satisfies the standard tissue contribution floor"
            if not applies or actual >= floor
            else (
                "burden edit equals the deterministically proven maximum-safe tissue capacity below the standard floor"
                if adaptive_fallback
                else "cell changes cannot substitute for required tissue burden"
            )
        ),
        metrics={
            "applies": applies,
            "actual_pixels": actual,
            "standard_floor_pixels": floor,
            "used_capacity_adaptive_fallback": adaptive_fallback,
        },
    )


def _cell_tissue_compatibility(c):
    source = c.source_nuclei
    target = c.candidate.target_nuclei_mask
    added = (target > 0) & (target != source)
    tumor = np.isin(c.candidate.target_tissue_mask, c.schema.tumor_fine_ids)
    neo = added & (target == 1)
    added_instance_classes = []
    for _, class_id, component in iter_instances(target):
        if np.any(component & added):
            added_instance_classes.append(int(class_id))
    illegal_classes = sorted(
        set(added_instance_classes) - set(c.plan.cell_plan.allowed_cell_classes)
    )
    incompatible_neo = int(np.count_nonzero(neo & ~tumor))
    if c.plan.coupling_plan.allow_neoplastic_in_non_tumor_tissue:
        allowed_halo = _authorized_cell_zone(c)
        incompatible_neo = int(np.count_nonzero(neo & ~tumor & ~allowed_halo))
    passed = incompatible_neo == 0 and not illegal_classes
    return _result(
        "cell_tissue_compatibility",
        passed,
        "new cell classes are compatible with target tissue/mechanism"
        if passed
        else "new cells use an unauthorized class or neoplastic pixels occur outside their authorized zone",
        metrics={
            "incompatible_neoplastic_pixels": incompatible_neo,
            "added_instance_classes": added_instance_classes,
            "allowed_cell_classes": list(c.plan.cell_plan.allowed_cell_classes),
            "illegal_added_classes": illegal_classes,
        },
    )


def _cell_zone_localization(c):
    cell_only = c.candidate.cell_change & ~c.candidate.tissue_change
    maximum = c.plan.coupling_plan.maximum_halo_px
    allowed = _authorized_cell_zone(c)
    violations = int(np.count_nonzero(cell_only & ~allowed))
    passed = violations == 0
    return _result("cell_zone_localization", passed, "cell-only changes remain inside the skill-authorized interface halo" if passed else "cell-only change exceeds its mechanism zone", metrics={"violation_pixels": violations, "maximum_halo_px": maximum})


def _joint_provenance(c):
    required = ("layout_tool_version", "ranker", "seed", "mechanism_id", "skill_version", "tissue_tool_trace", "compiled_cell_tool_program")
    missing = [item for item in required if item not in c.candidate.tool_trace]
    passed = not missing and not c.candidate.tool_trace.get("cross_domain_fallback", True)
    return _result("joint_provenance", passed, "joint candidate binds skill, tools, ranker and seed" if passed else f"missing/unsafe provenance={missing}", metrics={"missing": missing})


def _profile_provenance(c):
    required = c.bundle.annotation_profile.required_provenance_fields
    def known(value):
        if not value:
            return False
        if isinstance(value, str) and (
            value.lower().startswith("unknown") or "not_recorded" in value.lower()
        ):
            return False
        return True
    missing = [item for item in required if not known(c.case.provenance.get(item))]
    return _result("profile_provenance", not missing, "annotation-profile provenance is complete" if not missing else "missing profile provenance: " + ", ".join(missing), metrics={"required": list(required), "missing": missing})


def _prohibited_cell_region(c):
    ids = c.bundle.annotation_profile.prohibit_cell_placement_fine_ids
    added = (c.candidate.target_nuclei_mask > 0) & (c.candidate.target_nuclei_mask != c.source_nuclei)
    violations = int(np.count_nonzero(added & np.isin(c.candidate.target_tissue_mask, ids)))
    return _result("prohibited_cell_region", violations == 0, "no new nucleus enters a profile-prohibited region" if violations == 0 else "new nucleus overlaps prohibited region", metrics={"violation_pixels": violations})


def _prohibited_generation_support(c):
    ids = c.bundle.annotation_profile.prohibit_generation_support_fine_ids
    violations = int(np.count_nonzero(c.candidate.generation_support & np.isin(c.source_tissue, ids)))
    return _result("prohibited_generation_support", violations == 0, "generation support excludes profile-prohibited regions" if violations == 0 else "generation support enters prohibited region", metrics={"violation_pixels": violations})


def _orca_fragment_protection(c):
    changed = c.candidate.joint_change | c.candidate.generation_support
    background = np.isin(c.source_tissue, c.bundle.annotation_profile.prohibited_fine_ids)
    violations = int(np.count_nonzero(changed & background))
    return _result("orca_fragment_protection", violations == 0, "fragmented ORCA non-tissue is not an edit seed, cell zone or generation support" if violations == 0 else "ORCA fragmented non-tissue was entered", metrics={"violation_pixels": violations})


def _fine_pattern_preserved(c):
    source = c.source_tissue
    target = c.candidate.target_tissue_mask
    change = c.candidate.tissue_change
    tumor_to_other_tumor = change & np.isin(source, c.schema.tumor_fine_ids) & np.isin(target, c.schema.tumor_fine_ids) & (source != target)
    violations = int(np.count_nonzero(tumor_to_other_tumor))
    required = c.bundle.annotation_profile.mechanism_required_fine_ids.get(c.plan.selected_mechanism_id, ())
    source_fine = set(int(value) for value in np.unique(source[change]))
    pattern_mismatch = bool(required) and not source_fine.issubset(set(required))
    passed = violations == 0 and not pattern_mismatch
    return _result("fine_pattern_preserved", passed, "no Gleason fine ID was converted and edited source matches the selected pattern mechanism" if passed else "implicit fine-label conversion or pattern-mechanism mismatch", metrics={"violation_pixels": violations, "required_source_fine_ids": list(required), "observed_changed_source_fine_ids": sorted(source_fine), "pattern_mismatch": pattern_mismatch})


def _touches_border(component):
    return bool(np.any(component[0]) or np.any(component[-1]) or np.any(component[:, 0]) or np.any(component[:, -1]))


def _spatial_summary(mask):
    mask = np.asarray(mask)
    instances = tuple(iter_instances(mask))
    centers = []
    areas = []
    class_counts = {class_id: 0 for class_id in range(1, 6)}
    for _, class_id, component in instances:
        cy, cx = ndimage.center_of_mass(component)
        centers.append((float(cy), float(cx)))
        areas.append(int(np.count_nonzero(component)))
        class_counts[int(class_id)] = class_counts.get(int(class_id), 0) + 1
    mean_nnd = None
    ripley = None
    if len(centers) >= 2:
        points = np.asarray(centers, dtype=float)
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        mean_nnd = float(np.mean(distances[:, 1]))
        radius = 24.0
        unordered_pairs = len(tree.query_pairs(radius))
        area = float(np.prod(mask.shape))
        ripley_k = area * (2.0 * unordered_pairs) / (
            len(points) * (len(points) - 1)
        )
        ripley = float(ripley_k / (np.pi * radius * radius))
    return {
        "instance_count": len(instances),
        "density_per_kpx": float(len(instances) * 1000.0 / max(1, mask.size)),
        "mean_nnd_px": mean_nnd,
        "instance_area_p50": (float(np.percentile(areas, 50)) if areas else None),
        "instance_area_p95": (float(np.percentile(areas, 95)) if areas else None),
        "ripley_k24_normalized": ripley,
        "class_counts": class_counts,
    }


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _instance_centers_in_region(mask, region):
    count = 0
    for _, _, component in iter_instances(mask):
        row, col = ndimage.center_of_mass(component)
        row, col = int(round(row)), int(round(col))
        if (
            0 <= row < region.shape[0]
            and 0 <= col < region.shape[1]
            and region[row, col]
        ):
            count += 1
    return count


def _authorized_cell_zone(c):
    if c.bundle.primitive.scope == "tissue_and_cell":
        return ndimage.binary_dilation(
            c.candidate.tissue_change,
            iterations=c.plan.coupling_plan.maximum_halo_px,
        )
    interface_masks = [
        c.scene.tissue.interface_masks[interface_id]
        for interface_id in c.plan.cell_plan.interface_ids
        if interface_id in c.scene.tissue.interface_masks
    ]
    if not interface_masks:
        return np.zeros_like(c.candidate.cell_change, dtype=bool)
    interface = np.logical_or.reduce(interface_masks)
    distance = ndimage.distance_transform_edt(~interface)
    minimum = c.bundle.mechanism.cell_program.halo_distance_px[0]
    maximum = c.plan.coupling_plan.maximum_halo_px
    if c.case.cell_count_extent_budget is not None:
        minimum = max(minimum, c.case.cell_count_extent_budget.interface_min_px)
        maximum = min(
            maximum,
            c.case.cell_count_extent_budget.interface_max_px,
            c.case.cell_count_extent_budget.maximum_extent_px,
        )
    return (distance >= minimum) & (distance <= maximum)


def _maximum_changed_distance_to_interfaces(c):
    changed = np.asarray(c.candidate.cell_change, dtype=bool)
    interface_masks = [
        c.scene.tissue.interface_masks[interface_id]
        for interface_id in c.plan.cell_plan.interface_ids
        if interface_id in c.scene.tissue.interface_masks
    ]
    if not np.any(changed) or not interface_masks:
        return float("inf") if np.any(changed) else 0.0
    interface = np.logical_or.reduce(interface_masks)
    distance = ndimage.distance_transform_edt(~interface)
    return float(np.max(distance[changed]))


def _proven_max_safe_tissue_fallback(c):
    if c.case.joint_area_budget.capacity_floor_policy != "lower_to_proven_max_safe":
        return False
    trace = c.candidate.tool_trace.get("tissue_tool_trace", {})
    if not isinstance(trace, dict):
        return False
    resolved = int(trace.get("resolved_target_pixels", -1))
    desired = int(trace.get("desired_target_pixels", -1))
    actual = c.candidate.ledger.tissue_pixels
    return bool(
        trace.get("area_fallback_used") is True
        and 0 < resolved < desired
        and actual == resolved
    )

"""Deterministic enforcement for skill-owned Planner authority."""

from __future__ import annotations

from .models import JointContractError

PLANNER_OBSERVATION_SOURCES = frozenset(
    {
        "instruction",
        "semantic_intent",
        "tissue_mask",
        "nuclei_mask",
        "scene_graph",
        "candidate_certificate",
        "skill_rules",
        "user_roi",
        "auxiliary_masks",
    }
)
PLANNER_DECISIONS = frozenset(
    {
        "select_primitive_mechanism_pair",
        "select_certified_tissue_plan_candidate",
        "select_certified_cell_plan_candidate",
        "select_certified_interface_anchor_ids",
        "select_allowed_tool_program",
        "request_clarification",
        "abstain",
    }
)
PREFERENCE_METRIC_CATALOG = {
    "pref:broad-contact:minimize-depth-span-ratio": ("depth_span_ratio", "min"),
    "pref:packing-seam:maximize-capacity-margin": (
        "packing_seam_capacity_margin",
        "max",
    ),
    "pref:anchor-distribution:minimize-maximum-depth": ("maximum_depth_px", "min"),
    "pref:protected-clearance:minimize-exclusions": (
        "protected_exclusion_count",
        "min",
    ),
    "pref:directional-anchor:maximize-length-depth-ratio": (
        "anchor_length_depth_ratio",
        "max",
    ),
    "pref:class1-packing:maximize-capacity-margin": ("class1_packing_margin", "max"),
    "pref:projection:minimize-component-and-side-merge-count": (
        "projection_merge_count",
        "min",
    ),
    "pref:protected-clearance:maximize-distance": ("protected_distance_px", "max"),
    "pref:annulus:maximize-separated-focus-capacity": (
        "separated_focus_capacity",
        "max",
    ),
    "pref:annulus:minimize-distance-with-span-floor": (
        "median_tumor_distance_px",
        "min",
    ),
    "pref:complete-shape:maximize-packing-margin": (
        "complete_shape_packing_margin",
        "max",
    ),
    "pref:bridge:minimize-connectivity-risk": ("bridge_risk_count", "min"),
    "pref:certificate:maximize-capacity-margin": ("certificate_capacity_margin", "max"),
    "pref:topology:minimize-structural-risk": ("structural_risk_count", "min"),
}


def validate_planner_policy(policy) -> None:
    allowed = set(policy.allowed_observation_sources)
    prohibited = set(policy.prohibited_observation_sources)
    decisions = set(policy.allowed_decisions)
    preferences = set(policy.selection_preferences)
    if allowed - PLANNER_OBSERVATION_SOURCES or allowed & prohibited:
        raise JointContractError(
            "Planner policy contains an invalid observation authority"
        )
    if not {
        "source_he_for_execution",
        "unannotated_histology_inference",
    }.issubset(prohibited):
        raise JointContractError(
            "Planner policy does not prohibit raw H&E/unannotated inference"
        )
    if not decisions or decisions - PLANNER_DECISIONS:
        raise JointContractError("Planner policy contains an unsupported decision")
    if not preferences or preferences - set(PREFERENCE_METRIC_CATALOG):
        raise JointContractError(
            "Planner policy contains an unknown preference rule ID"
        )
    if not policy.hard_constraint_checker_ids:
        raise JointContractError("Planner policy omits hard checker bindings")


def preference_metadata(policy) -> tuple[dict[str, str], ...]:
    # The strict runtime validation is currently scoped to the revised Breast
    # execution surface.  Other organ skills retain legacy prose preferences
    # until their own refine cycle; exposing no metric binding is safer than
    # pretending that prose has become a deterministic candidate metric.
    if not policy.selection_preferences or any(
        rule_id not in PREFERENCE_METRIC_CATALOG
        for rule_id in policy.selection_preferences
    ):
        return ()
    return tuple(
        {
            "preference_rule_id": rule_id,
            "candidate_metric_id": PREFERENCE_METRIC_CATALOG[rule_id][0],
            "direction": PREFERENCE_METRIC_CATALOG[rule_id][1],
        }
        for rule_id in policy.selection_preferences
    )

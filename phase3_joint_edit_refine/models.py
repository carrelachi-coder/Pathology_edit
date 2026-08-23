"""Strict public contracts for atomic tissue--nuclei edits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phase3_mask_edit_refine.models import EditPlan


class JointContractError(ValueError):
    """Raised when a joint edit cannot be represented without guessing."""


CELL_ACTIONS = frozenset({"retain", "remove_whole", "add"})
CELL_BASELINE_MODES = frozenset(
    {
        "preserve",
        "regenerate_target_population",
        "selective_remove",
        "structured_add",
        "render_owned_clearance",
    }
)
CELL_QUOTA_ROLES = frozenset(
    {"within_total_quota", "explicit_increment", "explicit_decrement"}
)
LAYOUT_PROGRAMS = frozenset(
    {
        "preserve_only",
        "population_replacement",
        "single",
        "pair",
        "small_cluster",
        "short_cord",
        "boundary_aligned",
        "dense_sheet",
        "localized_density_gradient",
    }
)


@dataclass(frozen=True)
class JointAreaBudget:
    """Area contract for the union of tissue and complete-nucleus footprints."""

    target_fraction: float = 0.19
    min_fraction: float = 0.14
    max_fraction: float = 0.24
    tissue_min_fraction: float = 0.14
    basis: str = "whole_patch"
    relative_tolerance: float = 0.02
    fallback_policy: str = "max_feasible_below_target"
    capacity_floor_policy: str = "strict"
    minimum_effective_fraction: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.target_fraction,
            self.min_fraction,
            self.max_fraction,
            self.tissue_min_fraction,
        )
        if not all(isinstance(value, (int, float)) for value in values):
            raise JointContractError("joint area fractions must be numeric")
        if not 0.0 <= self.min_fraction <= self.target_fraction <= self.max_fraction <= 1.0:
            raise JointContractError(
                "joint area must satisfy 0 <= min <= target <= max <= 1"
            )
        if not 0.0 <= self.tissue_min_fraction <= self.target_fraction:
            raise JointContractError(
                "tissue_min_fraction must be inside [0, target_fraction]"
            )
        if not 0.0 <= self.minimum_effective_fraction <= self.tissue_min_fraction:
            raise JointContractError(
                "minimum_effective_fraction must be inside "
                "[0, tissue_min_fraction]"
            )
        if self.basis != "whole_patch":
            raise JointContractError("v1 joint area basis must be whole_patch")
        if not 0.0 <= self.relative_tolerance <= 0.25:
            raise JointContractError("relative_tolerance must be in [0, 0.25]")
        if self.fallback_policy not in {"exact", "max_feasible_below_target"}:
            raise JointContractError("unsupported joint area fallback policy")
        if self.capacity_floor_policy not in {
            "strict",
            "lower_to_proven_max_safe",
        }:
            raise JointContractError("unsupported capacity floor policy")
        if (
            self.capacity_floor_policy == "lower_to_proven_max_safe"
            and self.fallback_policy != "max_feasible_below_target"
        ):
            raise JointContractError(
                "capacity-adaptive floor requires max_feasible_below_target"
            )

    @classmethod
    def from_value(cls, value: Any) -> JointAreaBudget:
        if isinstance(value, (int, float)):
            target = float(value)
            return cls(
                target_fraction=target,
                min_fraction=target,
                max_fraction=target,
                tissue_min_fraction=0.0,
                fallback_policy="exact",
            )
        if not isinstance(value, Mapping):
            raise JointContractError("joint_area_budget must be numeric or a mapping")
        target = _number(value, "target_fraction")
        minimum = float(value.get("min_fraction", target))
        maximum = float(value.get("max_fraction", target))
        return cls(
            target_fraction=target,
            min_fraction=minimum,
            max_fraction=maximum,
            tissue_min_fraction=float(value.get("tissue_min_fraction", minimum)),
            basis=str(value.get("basis", "whole_patch")),
            relative_tolerance=float(value.get("relative_tolerance", 0.02)),
            fallback_policy=str(
                value.get(
                    "fallback_policy",
                    "max_feasible_below_target" if minimum < target else "exact",
                )
            ),
            capacity_floor_policy=str(
                value.get("capacity_floor_policy", "strict")
            ),
            minimum_effective_fraction=float(
                value.get("minimum_effective_fraction", 0.0)
            ),
        )

    def target_pixels(self, shape: tuple[int, int]) -> int:
        return round(int(np.prod(shape)) * self.target_fraction)

    def tissue_floor_pixels(self, shape: tuple[int, int]) -> int:
        return int(np.ceil(int(np.prod(shape)) * self.tissue_min_fraction))

    def tissue_execution_floor_pixels(self, shape: tuple[int, int]) -> int:
        """Compiler floor; the standard floor remains a downstream gate.

        A capacity-adaptive task may fall below its standard contribution
        floor only when the deterministic solver proves that it returned the
        maximum safe edit.  It must still clear the explicit meaningful-edit
        floor; this prevents visually negligible 1--2% fallbacks from being
        presented as successful burden edits.
        """

        if self.capacity_floor_policy == "lower_to_proven_max_safe":
            return int(
                np.ceil(
                    int(np.prod(shape)) * self.minimum_effective_fraction
                )
            )
        return self.tissue_floor_pixels(shape)

    def hard_interval_pixels(self, shape: tuple[int, int]) -> tuple[int, int]:
        total = int(np.prod(shape))
        return (
            int(np.ceil(total * self.min_fraction)),
            int(np.floor(total * self.max_fraction)),
        )

    def desired_interval_pixels(self, shape: tuple[int, int]) -> tuple[int, int]:
        target = self.target_pixels(shape)
        tolerance = max(1, int(np.ceil(target * self.relative_tolerance)))
        hard_min, hard_max = self.hard_interval_pixels(shape)
        return max(hard_min, target - tolerance), min(hard_max, target + tolerance)


@dataclass(frozen=True)
class CellCountExtentBudget:
    """Budget for a cell-only primitive; it never borrows the G2 tissue floor."""

    target_delta_count: int
    min_delta_count: int
    max_delta_count: int
    maximum_extent_px: int
    interface_min_px: int = 0
    interface_max_px: int = 48
    minimum_effect_span_px: int = 0
    minimum_effect_foci: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.min_delta_count <= self.target_delta_count <= self.max_delta_count:
            raise JointContractError(
                "cell count budget must satisfy 0 <= min <= target <= max"
            )
        if self.maximum_extent_px <= 0:
            raise JointContractError("cell-only maximum_extent_px must be positive")
        if not 0 <= self.interface_min_px <= self.interface_max_px:
            raise JointContractError("cell-only interface distance interval is invalid")
        if not 0 <= self.minimum_effect_span_px <= self.maximum_extent_px:
            raise JointContractError(
                "cell-only effect span must lie inside the maximum extent"
            )
        if not 0 <= self.minimum_effect_foci <= self.min_delta_count:
            raise JointContractError(
                "cell-only effect foci cannot exceed the minimum count"
            )

    @classmethod
    def from_value(cls, value: Any) -> CellCountExtentBudget | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise JointContractError("cell_count_extent_budget must be a mapping")
        target = int(value.get("target_delta_count", -1))
        return cls(
            target_delta_count=target,
            min_delta_count=int(value.get("min_delta_count", target)),
            max_delta_count=int(value.get("max_delta_count", target)),
            maximum_extent_px=int(value.get("maximum_extent_px", 48)),
            interface_min_px=int(value.get("interface_min_px", 0)),
            interface_max_px=int(value.get("interface_max_px", 48)),
            minimum_effect_span_px=int(
                value.get("minimum_effect_span_px", 0)
            ),
            minimum_effect_foci=int(value.get("minimum_effect_foci", 0)),
        )


@dataclass(frozen=True)
class JointCaseContext:
    """Four-axis identity and immutable source assets for a joint edit."""

    case_id: str
    instruction: str
    source_image_uri: str
    source_tissue_mask_uri: str
    source_nuclei_mask_uri: str
    pathology_domain_id: str
    annotation_profile_id: str
    cell_observation_profile_id: str
    cell_population_profile_id: str
    primitive_id: str
    joint_area_budget: JointAreaBudget | None
    seed: int
    provenance: dict[str, Any]
    source_nuclei_instances_uri: str | None = None
    auxiliary_structure_uris: dict[str, str] = field(default_factory=dict)
    pixel_size_um: float | None = None
    cell_count_extent_budget: CellCountExtentBudget | None = None
    semantic_intent: dict[str, Any] = field(default_factory=dict)
    clarification_decision: dict[str, Any] = field(default_factory=dict)

    def compiled_normalized_intent(self) -> str:
        """Return the parser-owned intent used by all downstream planners.

        The mask-graph Planner is allowed to select a pathology mechanism, but it
        must never reinterpret the user's requested primitive.  Older research
        fixtures without a semantic-intent ledger retain their literal
        instruction for backward-compatible, non-production use.
        """

        if not self.semantic_intent:
            return self.instruction
        direction = self.semantic_intent.get("direction")
        subject = self.semantic_intent.get("subject")
        if not isinstance(direction, str) or not direction.strip():
            raise JointContractError("semantic intent direction is missing")
        if not isinstance(subject, str) or not subject.strip():
            raise JointContractError("semantic intent subject is missing")
        parts = [direction.strip(), subject.strip()]
        cell_class = self.semantic_intent.get("explicit_cell_class")
        location = self.semantic_intent.get("explicit_location")
        if isinstance(cell_class, str) and cell_class.strip():
            parts.append(f"cell_class={cell_class.strip()}")
        if isinstance(location, str) and location.strip():
            parts.append(f"location={location.strip()}")
        return "; ".join(parts)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> JointCaseContext:
        if not isinstance(payload, Mapping):
            raise JointContractError("JointCaseContext payload must be a mapping")
        string_keys = (
            "case_id",
            "instruction",
            "source_image_uri",
            "source_tissue_mask_uri",
            "source_nuclei_mask_uri",
            "pathology_domain_id",
            "annotation_profile_id",
            "cell_observation_profile_id",
            "cell_population_profile_id",
            "primitive_id",
        )
        values = {key: _string(payload, key) for key in string_keys}
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise JointContractError("provenance must be a non-empty mapping")
        required_digests = (
            "source_image_sha256",
            "source_tissue_mask_sha256",
            "source_nuclei_mask_sha256",
        )
        missing = [key for key in required_digests if not provenance.get(key)]
        if missing:
            raise JointContractError(
                "joint provenance missing source digests: " + ", ".join(missing)
            )
        seed = payload.get("seed")
        if not isinstance(seed, int):
            raise JointContractError("seed must be an integer")
        instances_uri = payload.get("source_nuclei_instances_uri")
        if instances_uri is not None and (
            not isinstance(instances_uri, str) or not instances_uri.strip()
        ):
            raise JointContractError(
                "source_nuclei_instances_uri must be a non-empty string when provided"
            )
        if instances_uri and not provenance.get("source_nuclei_instances_sha256"):
            raise JointContractError(
                "native nucleus instances require source_nuclei_instances_sha256"
            )
        raw_auxiliary = payload.get("auxiliary_structure_uris", {})
        if not isinstance(raw_auxiliary, Mapping) or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in raw_auxiliary.items()
        ):
            raise JointContractError(
                "auxiliary_structure_uris must map structure IDs to non-empty paths"
            )
        auxiliary = {
            str(key).strip(): str(value).strip()
            for key, value in raw_auxiliary.items()
        }
        auxiliary_digests = provenance.get("auxiliary_structure_sha256", {})
        auxiliary_provenance = provenance.get(
            "auxiliary_structure_provenance", {}
        )
        if auxiliary and (
            not isinstance(auxiliary_digests, Mapping)
            or set(auxiliary) != set(auxiliary_digests)
            or not all(isinstance(value, str) and value for value in auxiliary_digests.values())
            or not isinstance(auxiliary_provenance, Mapping)
            or set(auxiliary) != set(auxiliary_provenance)
            or not all(
                isinstance(value, Mapping)
                and value.get("producer_id")
                and value.get("producer_version")
                and value.get("source_tissue_mask_sha256")
                == provenance.get("source_tissue_mask_sha256")
                and value.get("output_sha256")
                == auxiliary_digests.get(key)
                for key, value in auxiliary_provenance.items()
            )
        ):
            raise JointContractError(
                "each auxiliary structure requires a bound digest and producer provenance"
            )
        pixel_size = payload.get("pixel_size_um")
        if pixel_size is not None and (
            not isinstance(pixel_size, (int, float)) or float(pixel_size) <= 0
        ):
            raise JointContractError("pixel_size_um must be positive when provided")
        cell_budget = CellCountExtentBudget.from_value(
            payload.get("cell_count_extent_budget")
        )
        raw_clarification = payload.get("clarification_decision", {})
        if not isinstance(raw_clarification, Mapping):
            raise JointContractError("clarification_decision must be an object")
        return cls(
            **values,
            joint_area_budget=(
                JointAreaBudget.from_value(payload.get("joint_area_budget"))
                if payload.get("joint_area_budget") is not None
                else None
            ),
            seed=seed,
            provenance=dict(provenance),
            source_nuclei_instances_uri=instances_uri,
            auxiliary_structure_uris=auxiliary,
            pixel_size_um=float(pixel_size) if pixel_size is not None else None,
            cell_count_extent_budget=cell_budget,
            semantic_intent=(
                dict(payload.get("semantic_intent", {}))
                if isinstance(payload.get("semantic_intent", {}), Mapping)
                else {}
            ),
            clarification_decision=dict(raw_clarification),
        )

    def validate_local_inputs(self) -> None:
        if self.semantic_intent:
            raw_hypotheses = self.semantic_intent.get(
                "primitive_hypotheses", ()
            )
            if not isinstance(raw_hypotheses, (list, tuple)):
                raise JointContractError(
                    "semantic primitive hypotheses must be a sequence"
                )
            candidate_ids = {
                str(item.get("primitive_id"))
                for item in raw_hypotheses
                if isinstance(item, Mapping) and item.get("primitive_id")
            }
            selected = self.semantic_intent.get("selected_primitive_id")
            if (
                self.semantic_intent.get("schema_version")
                not in {
                    "joint-semantic-intent-v2",
                    "joint-semantic-intent-v3",
                }
                or self.semantic_intent.get("instruction") != self.instruction
                or not candidate_ids
                or self.primitive_id not in candidate_ids
                or (selected is not None and selected != self.primitive_id)
            ):
                raise JointContractError(
                    "semantic intent is not bound to this instruction and its primitive hypotheses"
                )
        paths = {
            "source_image_uri": self.source_image_uri,
            "source_tissue_mask_uri": self.source_tissue_mask_uri,
            "source_nuclei_mask_uri": self.source_nuclei_mask_uri,
        }
        if self.source_nuclei_instances_uri:
            paths["source_nuclei_instances_uri"] = self.source_nuclei_instances_uri
        paths.update(
            {
                f"auxiliary_structure_uris.{key}": value
                for key, value in self.auxiliary_structure_uris.items()
            }
        )
        for label, value in paths.items():
            if "://" in value and not value.startswith("file://"):
                continue
            path = Path(value.removeprefix("file://"))
            if not path.is_file():
                raise JointContractError(f"{label} does not exist: {path}")

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NucleusInstance:
    instance_id: str
    class_id: int
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    tissue_fine_id: int
    touches_border: bool
    source: str = "semantic_component"
    tissue_component_id: str | None = None
    nearest_interface_id: str | None = None
    distance_to_interface_px: float | None = None
    perimeter_px: float | None = None
    solidity: float | None = None
    eccentricity: float | None = None
    completeness_status: str = "complete"
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CellGraphEdge:
    source_instance_id: str
    target_instance_id: str
    relation: str
    distance_px: float
    same_class: bool
    same_tissue_component: bool


@dataclass(frozen=True)
class PopulationZone:
    zone_id: str
    zone_kind: str
    tissue_component_id: str | None
    interface_id: str | None
    side: str | None
    distance_band_px: tuple[float, float] | None
    area_px: int
    nucleus_count: int
    density_per_10k_px: float
    class_counts: dict[int, int]
    class_density_per_10k_px: dict[int, float]
    nucleus_area_quantiles: dict[str, float]
    nearest_neighbor_quantiles: dict[str, float]
    observation_quality: str


@dataclass(frozen=True)
class PopulationGraph:
    zones: tuple[PopulationZone, ...]
    adjacency: tuple[tuple[str, str], ...]
    median_nucleus_area_px: float | None
    nominal_nucleus_diameter_px: float | None
    warnings: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CellSceneGraph:
    width: int
    height: int
    instances: tuple[NucleusInstance, ...]
    class_counts: dict[int, int]
    mean_nearest_neighbor_px: float | None
    observation_quality: str
    warnings: tuple[str, ...] = ()
    edges: tuple[CellGraphEdge, ...] = ()
    interface_relation_count: int = 0
    merged_suspect_instance_ids: tuple[str, ...] = ()
    border_censored_instance_ids: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CellEditPlan:
    """Planner-selected cell intent; exact coordinates/counts are compiler-owned."""

    core_zone: str
    halo_zone: str | None
    actions: tuple[str, ...]
    allowed_cell_classes: tuple[int, ...]
    layout_program_id: str
    protected_instance_ids: tuple[str, ...]
    supporting_rule_ids: tuple[str, ...]
    expected_morphology: str
    baseline_mode: str = "regenerate_target_population"
    interface_ids: tuple[str, ...] = ()
    anchor_ids: tuple[str, ...] = ()
    spatial_anchor_type: str = "not_applicable"
    spatial_anchor_observation: str | None = None
    population_contract_id: str = "patch-adaptive-target-population-v1"
    mechanism_program_id: str = "population_replacement"
    mechanism_quota_role: str = "within_total_quota"
    tool_program_id: str = "joint-cell-tool-program-v2"
    erasure_policy: str = "complete-instance-intersection"
    placement_center_policy: str = "compiled-zone-centers"
    valid_footprint_policy: str = "target-tissue-full-containment"
    probnet_context_policy: str = "expanded-support-context-cleared"

    def __post_init__(self) -> None:
        if not self.core_zone:
            raise JointContractError("cell plan core_zone is required")
        if not self.actions or set(self.actions) - CELL_ACTIONS:
            raise JointContractError("cell plan contains unsupported or empty actions")
        if self.layout_program_id not in LAYOUT_PROGRAMS:
            raise JointContractError(
                f"unsupported cell layout program: {self.layout_program_id}"
            )
        if self.baseline_mode not in CELL_BASELINE_MODES:
            raise JointContractError(
                f"unsupported cell baseline mode: {self.baseline_mode}"
            )
        if self.mechanism_quota_role not in CELL_QUOTA_ROLES:
            raise JointContractError(
                f"unsupported mechanism quota role: {self.mechanism_quota_role}"
            )
        if self.mechanism_program_id not in LAYOUT_PROGRAMS:
            raise JointContractError(
                f"unsupported mechanism program: {self.mechanism_program_id}"
            )
        if any(value not in range(1, 6) for value in self.allowed_cell_classes):
            raise JointContractError("allowed cell classes must use internal IDs 1..5")
        if not self.supporting_rule_ids:
            raise JointContractError("cell plan must cite supporting joint rules")
        if self.spatial_anchor_type not in {
            "not_applicable",
            "interface",
            "population_peak",
        }:
            raise JointContractError("unsupported cell spatial anchor type")
        if self.spatial_anchor_type == "interface":
            if not self.interface_ids or not self.anchor_ids:
                raise JointContractError(
                    "interface-anchored cell plan requires interface and anchor IDs"
                )
            if not self.spatial_anchor_observation:
                raise JointContractError(
                    "interface-anchored cell plan requires a visible observation"
                )
        if self.spatial_anchor_type == "population_peak":
            if self.interface_ids or self.anchor_ids:
                raise JointContractError(
                    "population-peak cell plan cannot claim interface anchors"
                )
            if not self.spatial_anchor_observation:
                raise JointContractError(
                    "population-peak cell plan requires a mask-derived observation"
                )

    @classmethod
    def from_mapping(cls, payload: Any) -> CellEditPlan:
        if not isinstance(payload, Mapping):
            raise JointContractError("cell_plan is required and must be a mapping")
        return cls(
            core_zone=_string(payload, "core_zone"),
            halo_zone=_optional_string(payload.get("halo_zone")),
            actions=_string_tuple(payload.get("actions"), "cell_plan.actions"),
            allowed_cell_classes=_int_tuple(
                payload.get("allowed_cell_classes", ()),
                "cell_plan.allowed_cell_classes",
            ),
            layout_program_id=_string(payload, "layout_program_id"),
            protected_instance_ids=_string_tuple(
                payload.get("protected_instance_ids", ()),
                "cell_plan.protected_instance_ids",
                allow_empty=True,
            ),
            supporting_rule_ids=_string_tuple(
                payload.get("supporting_rule_ids"),
                "cell_plan.supporting_rule_ids",
            ),
            expected_morphology=_string(payload, "expected_morphology"),
            baseline_mode=str(
                payload.get("baseline_mode", "regenerate_target_population")
            ),
            interface_ids=_string_tuple(
                payload.get("interface_ids", ()),
                "cell_plan.interface_ids",
                allow_empty=True,
            ),
            anchor_ids=_string_tuple(
                payload.get("anchor_ids", ()),
                "cell_plan.anchor_ids",
                allow_empty=True,
            ),
            spatial_anchor_type=str(
                payload.get("spatial_anchor_type", "not_applicable")
            ),
            spatial_anchor_observation=_optional_string(
                payload.get("spatial_anchor_observation")
            ),
            population_contract_id=str(
                payload.get(
                    "population_contract_id",
                    "patch-adaptive-target-population-v1",
                )
            ),
            mechanism_program_id=str(
                payload.get(
                    "mechanism_program_id",
                    payload.get("layout_program_id", "population_replacement"),
                )
            ),
            mechanism_quota_role=str(
                payload.get("mechanism_quota_role", "within_total_quota")
            ),
            tool_program_id=str(
                payload.get("tool_program_id", "joint-cell-tool-program-v2")
            ),
            erasure_policy=str(
                payload.get("erasure_policy", "complete-instance-intersection")
            ),
            placement_center_policy=str(
                payload.get("placement_center_policy", "compiled-zone-centers")
            ),
            valid_footprint_policy=str(
                payload.get(
                    "valid_footprint_policy", "target-tissue-full-containment"
                )
            ),
            probnet_context_policy=str(
                payload.get(
                    "probnet_context_policy",
                    "expanded-support-context-cleared",
                )
            ),
        )


@dataclass(frozen=True)
class CouplingPlan:
    compatibility_rule_ids: tuple[str, ...]
    area_contract_id: str
    render_support_policy_id: str
    allow_neoplastic_in_non_tumor_tissue: bool = False
    maximum_halo_px: int = 24

    def __post_init__(self) -> None:
        if not self.compatibility_rule_ids:
            raise JointContractError("coupling plan must cite compatibility rules")
        if not self.area_contract_id or not self.render_support_policy_id:
            raise JointContractError("coupling plan area/render policies are required")
        if not 0 <= self.maximum_halo_px <= 128:
            raise JointContractError("maximum_halo_px must be in [0, 128]")

    @classmethod
    def from_mapping(cls, payload: Any) -> CouplingPlan:
        if not isinstance(payload, Mapping):
            raise JointContractError("coupling_plan is required and must be a mapping")
        return cls(
            compatibility_rule_ids=_string_tuple(
                payload.get("compatibility_rule_ids"),
                "coupling_plan.compatibility_rule_ids",
            ),
            area_contract_id=_string(payload, "area_contract_id"),
            render_support_policy_id=_string(payload, "render_support_policy_id"),
            allow_neoplastic_in_non_tumor_tissue=bool(
                payload.get("allow_neoplastic_in_non_tumor_tissue", False)
            ),
            maximum_halo_px=int(payload.get("maximum_halo_px", 24)),
        )


@dataclass(frozen=True)
class JointEditPlan:
    schema_version: str
    case_id: str
    normalized_intent: str
    selected_mechanism_id: str
    supporting_observations: tuple[str, ...]
    supporting_rule_ids: tuple[str, ...]
    representability_confidence: float
    tissue_plan: EditPlan | None
    cell_plan: CellEditPlan
    coupling_plan: CouplingPlan
    uncertainties: tuple[str, ...]
    escalation_reason: str | None = None
    structural_unit_ids: tuple[str, ...] = ()
    supporting_preference_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tissue_plan is not None and self.tissue_plan.case_id != self.case_id:
            raise JointContractError("tissue plan case_id does not match joint plan")
        if not self.cell_plan:
            raise JointContractError("every joint plan requires a cell plan")
        if not 0.0 <= self.representability_confidence <= 1.0:
            raise JointContractError("representability confidence must be in [0, 1]")
        if not self.supporting_observations or not self.supporting_rule_ids:
            raise JointContractError("joint plan must cite observations and rules")
        if len(set(self.structural_unit_ids)) != len(self.structural_unit_ids):
            raise JointContractError("joint plan structural unit IDs must be unique")
        if len(set(self.supporting_preference_rule_ids)) != len(
            self.supporting_preference_rule_ids
        ):
            raise JointContractError(
                "joint plan preference rule IDs must be unique"
            )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChangeLedger:
    tissue_pixels: int
    removed_nucleus_pixels: int
    added_nucleus_pixels: int
    cell_pixels: int
    cell_only_pixels: int
    joint_pixels: int
    generation_support_pixels: int
    total_pixels: int
    removed_instance_ids: tuple[str, ...]
    added_instance_ids: tuple[str, ...]
    retained_instance_ids: tuple[str, ...]

    @property
    def tissue_fraction(self) -> float:
        return self.tissue_pixels / max(1, self.total_pixels)

    @property
    def cell_fraction(self) -> float:
        return self.cell_pixels / max(1, self.total_pixels)

    @property
    def joint_fraction(self) -> float:
        return self.joint_pixels / max(1, self.total_pixels)

    @property
    def generation_support_fraction(self) -> float:
        return self.generation_support_pixels / max(1, self.total_pixels)

    def to_metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "tissue_fraction": self.tissue_fraction,
            "cell_fraction": self.cell_fraction,
            "joint_fraction": self.joint_fraction,
            "generation_support_fraction": self.generation_support_fraction,
        }


@dataclass(frozen=True)
class JointCandidate:
    candidate_id: str
    tissue_candidate_id: str
    cell_candidate_id: str
    mechanism_id: str
    target_tissue_mask: np.ndarray
    target_nuclei_mask: np.ndarray
    tissue_change: np.ndarray
    cell_change: np.ndarray
    joint_change: np.ndarray
    generation_support: np.ndarray
    ledger: ChangeLedger
    tool_trace: dict[str, Any]

    def __post_init__(self) -> None:
        arrays = (
            self.target_tissue_mask,
            self.target_nuclei_mask,
            self.tissue_change,
            self.cell_change,
            self.joint_change,
            self.generation_support,
        )
        if any(np.asarray(item).ndim != 2 for item in arrays):
            raise JointContractError("joint candidate arrays must be 2-D")
        shape = np.asarray(self.target_tissue_mask).shape
        if any(np.asarray(item).shape != shape for item in arrays[1:]):
            raise JointContractError("joint candidate arrays must share one shape")
        if np.any(np.asarray(self.tissue_change, dtype=bool) & ~self.joint_change):
            raise JointContractError("joint change must contain tissue change")
        if np.any(np.asarray(self.cell_change, dtype=bool) & ~self.joint_change):
            raise JointContractError("joint change must contain cell change")
        if np.any(np.asarray(self.joint_change, dtype=bool) & ~self.generation_support):
            raise JointContractError("generation support must contain joint change")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "tissue_candidate_id": self.tissue_candidate_id,
            "cell_candidate_id": self.cell_candidate_id,
            "mechanism_id": self.mechanism_id,
            "ledger": self.ledger.to_metadata(),
            "tool_trace": dict(self.tool_trace),
        }


@dataclass(frozen=True)
class JointGateCheck:
    check_id: str
    passed: bool
    severity: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JointGateReport:
    candidate_id: str
    passed: bool
    checks: tuple[JointGateCheck, ...]

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointCriticRanking:
    candidate_id: str
    score: float
    confidence: float
    supporting_rule_ids: tuple[str, ...]
    veto_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointCriticResult:
    rankings: tuple[JointCriticRanking, ...]
    abstain: bool
    summary: str
    usage: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointCondition:
    case_id: str
    candidate_id: str
    executable_contract_id: str
    target_tissue_mask: np.ndarray
    target_nuclei_mask: np.ndarray
    tissue_change: np.ndarray
    cell_change: np.ndarray
    joint_change: np.ndarray
    generation_support: np.ndarray
    pathology_mechanism: str
    active_skill_rules: tuple[str, ...]
    ledger: ChangeLedger


@dataclass(frozen=True)
class JointWorkflowResult:
    status: str
    case_context: JointCaseContext
    joint_plan: JointEditPlan | None
    gate_reports: tuple[JointGateReport, ...]
    critic_result: JointCriticResult | None
    selected_candidate_id: str | None
    condition: JointCondition | None
    abstain_reasons: tuple[str, ...]
    artifact_paths: dict[str, str]
    clarification_request: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JointContractError(f"{key} is required and must be a non-empty string")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JointContractError("optional strings must be non-empty when provided")
    return value.strip()


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise JointContractError(f"{key} is required and must be numeric")
    return float(value)


def _string_tuple(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise JointContractError(f"{label} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if (not result and not allow_empty) or len(result) != len(value):
        raise JointContractError(f"{label} contains empty values")
    return result


def _int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise JointContractError(f"{label} must be a sequence of integers")
    result = tuple(int(item) for item in value)
    if any(isinstance(item, bool) for item in value):
        raise JointContractError(f"{label} must contain integers")
    return result

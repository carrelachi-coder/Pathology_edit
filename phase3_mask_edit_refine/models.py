"""Public data contracts for the mask-edit-refine pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class RefineContractError(ValueError):
    """Raised when an input or model response violates a public contract."""


@dataclass(frozen=True)
class AreaBudget:
    """Explicit changed-area contract; the LLM may not modify it."""

    target_fraction: float
    min_fraction: float
    max_fraction: float
    basis: str = "whole_mask"
    relative_tolerance: float = 0.02
    fallback_policy: str = "exact"

    def __post_init__(self) -> None:
        values = (self.target_fraction, self.min_fraction, self.max_fraction)
        if not all(isinstance(value, (int, float)) for value in values):
            raise RefineContractError("area budget fractions must be numeric")
        if not 0.0 <= self.min_fraction <= self.target_fraction <= self.max_fraction <= 1.0:
            raise RefineContractError(
                "area budget must satisfy 0 <= min <= target <= max <= 1"
            )
        if self.basis not in {"whole_mask", "source_labels"}:
            raise RefineContractError(
                "area budget basis must be 'whole_mask' or 'source_labels'"
            )
        if not 0.0 <= float(self.relative_tolerance) <= 0.25:
            raise RefineContractError("relative_tolerance must be in [0, 0.25]")
        if self.fallback_policy not in {"exact", "max_feasible_below_target"}:
            raise RefineContractError(
                "fallback_policy must be 'exact' or 'max_feasible_below_target'"
            )
        if self.fallback_policy == "max_feasible_below_target" and (
            self.min_fraction >= self.target_fraction
        ):
            raise RefineContractError(
                "max_feasible_below_target requires min_fraction < target_fraction"
            )

    @classmethod
    def from_value(cls, value: Any) -> AreaBudget:
        if isinstance(value, (int, float)):
            target = float(value)
            return cls(target, target, target)
        if not isinstance(value, Mapping):
            raise RefineContractError("area_budget is required and must be numeric or a mapping")
        target = _required_number(value, "target_fraction")
        minimum = float(value.get("min_fraction", target))
        fallback = value.get("fallback_policy")
        if fallback is None:
            # A non-degenerate lower bound declares that the task itself allows
            # a smaller edit. Preserve legacy exact behavior when no such range
            # exists.
            fallback = (
                "max_feasible_below_target" if minimum < target else "exact"
            )
        return cls(
            target_fraction=target,
            min_fraction=minimum,
            max_fraction=float(value.get("max_fraction", target)),
            basis=str(value.get("basis", "whole_mask")),
            relative_tolerance=float(value.get("relative_tolerance", 0.02)),
            fallback_policy=str(fallback),
        )

    def denominator_pixels(self, mask: np.ndarray, source_region: np.ndarray) -> int:
        if self.basis == "source_labels":
            return int(np.count_nonzero(source_region))
        return int(np.asarray(mask).size)

    def target_pixels(self, mask: np.ndarray, source_region: np.ndarray) -> int:
        denominator = self.denominator_pixels(mask, source_region)
        return max(0, round(denominator * self.target_fraction))

    def allowed_pixel_interval(
        self, mask: np.ndarray, source_region: np.ndarray
    ) -> tuple[int, int]:
        denominator = self.denominator_pixels(mask, source_region)
        target = self.target_pixels(mask, source_region)
        interval_min = int(np.floor(denominator * self.min_fraction))
        interval_max = int(np.ceil(denominator * self.max_fraction))
        tolerance = max(1, int(np.ceil(target * self.relative_tolerance)))
        return min(interval_min, target - tolerance), max(interval_max, target + tolerance)

    def hard_pixel_interval(
        self, mask: np.ndarray, source_region: np.ndarray
    ) -> tuple[int, int]:
        """Return the task-declared hard range without target tolerance."""

        denominator = self.denominator_pixels(mask, source_region)
        if np.isclose(self.min_fraction, self.max_fraction, rtol=0.0, atol=1e-12):
            exact = max(0, round(denominator * self.target_fraction))
            return exact, exact
        return (
            max(0, int(np.ceil(denominator * self.min_fraction))),
            max(0, int(np.floor(denominator * self.max_fraction))),
        )


@dataclass(frozen=True)
class ResolvedAreaContract:
    """Deterministic resolution of a desired area into executable pixels."""

    desired_pixels: int
    hard_min_pixels: int
    hard_max_pixels: int
    resolved_pixels: int
    fallback_policy: str
    used_fallback: bool
    binding_constraint: str
    solver_version: str

    def __post_init__(self) -> None:
        if not (
            0 <= self.hard_min_pixels
            <= self.resolved_pixels
            <= self.hard_max_pixels
        ):
            raise RefineContractError(
                "resolved area must remain inside the hard pixel interval"
            )
        if self.resolved_pixels > self.desired_pixels:
            raise RefineContractError("resolved area may not exceed desired pixels")
        if self.used_fallback != (self.resolved_pixels < self.desired_pixels):
            raise RefineContractError("resolved area fallback flag is inconsistent")

    @classmethod
    def from_mapping(cls, payload: Any) -> ResolvedAreaContract | None:
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise RefineContractError("resolved_area must be a mapping or null")
        required = (
            "desired_pixels",
            "hard_min_pixels",
            "hard_max_pixels",
            "resolved_pixels",
            "fallback_policy",
            "used_fallback",
            "binding_constraint",
            "solver_version",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise RefineContractError(
                "resolved_area missing fields: " + ", ".join(missing)
            )
        return cls(
            desired_pixels=int(payload["desired_pixels"]),
            hard_min_pixels=int(payload["hard_min_pixels"]),
            hard_max_pixels=int(payload["hard_max_pixels"]),
            resolved_pixels=int(payload["resolved_pixels"]),
            fallback_policy=str(payload["fallback_policy"]),
            used_fallback=bool(payload["used_fallback"]),
            binding_constraint=str(payload["binding_constraint"]),
            solver_version=str(payload["solver_version"]),
        )


@dataclass(frozen=True)
class CaseContext:
    """Required dual-axis case identity and immutable edit request."""

    case_id: str
    instruction: str
    source_image_uri: str
    source_mask_uri: str
    pathology_domain_id: str
    annotation_profile_id: str
    primitive_id: str
    area_budget: AreaBudget
    seed: int
    provenance: dict[str, Any]
    pixel_size_um: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CaseContext:
        if not isinstance(payload, Mapping):
            raise RefineContractError("CaseContext payload must be a mapping")
        forbidden_legacy_keys = sorted(
            key for key in ("dataset", "reference_profile", "mask_profile") if key in payload
        )
        if forbidden_legacy_keys:
            raise RefineContractError(
                "legacy profile inference fields are forbidden; use annotation_profile_id: "
                + ", ".join(forbidden_legacy_keys)
            )
        missing = [
            key
            for key in (
                "case_id",
                "instruction",
                "source_image_uri",
                "source_mask_uri",
                "pathology_domain_id",
                "annotation_profile_id",
                "primitive_id",
                "area_budget",
                "seed",
                "provenance",
            )
            if key not in payload
        ]
        if missing:
            raise RefineContractError(
                "CaseContext missing required fields: " + ", ".join(missing)
            )
        strings = {
            key: _required_string(payload, key)
            for key in (
                "case_id",
                "instruction",
                "source_image_uri",
                "source_mask_uri",
                "pathology_domain_id",
                "annotation_profile_id",
                "primitive_id",
            )
        }
        provenance = payload["provenance"]
        if not isinstance(provenance, Mapping) or not provenance:
            raise RefineContractError("provenance must be a non-empty mapping")
        missing_digests = [
            key
            for key in ("source_image_sha256", "source_mask_sha256")
            if not isinstance(provenance.get(key), str) or not provenance.get(key)
        ]
        if missing_digests:
            raise RefineContractError(
                "provenance missing required source digests: " + ", ".join(missing_digests)
            )
        seed = payload["seed"]
        if not isinstance(seed, int):
            raise RefineContractError("seed must be an integer")
        pixel_size = payload.get("pixel_size_um")
        if pixel_size is not None and (
            not isinstance(pixel_size, (int, float)) or float(pixel_size) <= 0
        ):
            raise RefineContractError("pixel_size_um must be positive when provided")
        return cls(
            **strings,
            area_budget=AreaBudget.from_value(payload["area_budget"]),
            seed=seed,
            provenance=dict(provenance),
            pixel_size_um=float(pixel_size) if pixel_size is not None else None,
        )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    def validate_local_inputs(self) -> None:
        for key, uri in (
            ("source_image_uri", self.source_image_uri),
            ("source_mask_uri", self.source_mask_uri),
        ):
            if "://" in uri and not uri.startswith("file://"):
                continue
            path = Path(uri.removeprefix("file://"))
            if not path.is_file():
                raise RefineContractError(f"{key} does not exist: {path}")


@dataclass(frozen=True)
class SceneComponent:
    component_id: str
    label: str
    fine_ids: tuple[int, ...]
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    touches_border: bool


@dataclass(frozen=True)
class SceneInterface:
    interface_id: str
    source_component_id: str
    target_component_id: str
    source_label: str
    target_label: str
    contact_pixels: int
    bbox_xyxy: tuple[int, int, int, int]
    anchor_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class SceneAnchorSegment:
    """A deterministic, selectable sub-arc of one directed interface."""

    anchor_segment_id: str
    interface_id: str
    display_index: int
    contact_pixels: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]


@dataclass(frozen=True)
class SceneGraph:
    width: int
    height: int
    labels_present: dict[str, int]
    components: tuple[SceneComponent, ...]
    interfaces: tuple[SceneInterface, ...]
    anchor_segments: tuple[SceneAnchorSegment, ...]
    pixel_size_um: float | None
    warnings: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthProfile:
    """Executable front-depth intent along the selected interface anchors."""

    mode: str
    peak_depth_px: float
    edge_depth_px: float
    taper_fraction: float
    lobe_count: int
    noise_amplitude_px: float
    noise_correlation_px: float

    @classmethod
    def from_mapping(cls, payload: Any, *, key: str) -> DepthProfile:
        if not isinstance(payload, Mapping):
            raise RefineContractError(f"{key} must be a mapping")
        mode = _required_string(payload, "mode")
        if mode not in {"tapered_lobe", "uniform_front", "multi_lobe"}:
            raise RefineContractError(f"{key}.mode is unsupported: {mode}")
        peak = _required_number(payload, "peak_depth_px")
        edge = _required_number(payload, "edge_depth_px")
        taper = _required_number(payload, "taper_fraction")
        lobe_count = payload.get("lobe_count")
        noise_amplitude = _required_number(payload, "noise_amplitude_px")
        noise_correlation = _required_number(payload, "noise_correlation_px")
        if peak <= 0 or edge < 0 or edge > peak:
            raise RefineContractError(
                f"{key} requires 0 <= edge_depth_px <= peak_depth_px"
            )
        if not 0.0 <= taper <= 0.5:
            raise RefineContractError(f"{key}.taper_fraction must be in [0, 0.5]")
        if not isinstance(lobe_count, int) or not 1 <= lobe_count <= 3:
            raise RefineContractError(f"{key}.lobe_count must be an integer in [1, 3]")
        if not 0.0 <= noise_amplitude <= peak:
            raise RefineContractError(
                f"{key}.noise_amplitude_px must be in [0, peak_depth_px]"
            )
        if noise_correlation <= 0:
            raise RefineContractError(f"{key}.noise_correlation_px must be positive")
        return cls(
            mode=mode,
            peak_depth_px=peak,
            edge_depth_px=edge,
            taper_fraction=taper,
            lobe_count=lobe_count,
            noise_amplitude_px=noise_amplitude,
            noise_correlation_px=noise_correlation,
        )


@dataclass(frozen=True)
class InterfaceExecutionContract:
    """Pixel-executable contract compiled from one Planner interface choice."""

    anchor_segment_ids: tuple[str, ...]
    area_allocation_fraction: float
    depth_profile: DepthProfile
    min_anchor_coverage_fraction: float
    max_off_anchor_contact_fraction: float
    allocation_tolerance_fraction: float

    @classmethod
    def from_mapping(
        cls, payload: Any, *, key: str
    ) -> InterfaceExecutionContract:
        if not isinstance(payload, Mapping):
            raise RefineContractError(f"{key} must be a mapping")
        anchors = _string_tuple(
            payload.get("anchor_segment_ids"), f"{key}.anchor_segment_ids"
        )
        if not anchors:
            raise RefineContractError(f"{key}.anchor_segment_ids must not be empty")
        allocation = _required_number(payload, "area_allocation_fraction")
        coverage = _required_number(payload, "min_anchor_coverage_fraction")
        off_anchor = _required_number(payload, "max_off_anchor_contact_fraction")
        tolerance = _required_number(payload, "allocation_tolerance_fraction")
        if not 0.0 < allocation <= 1.0:
            raise RefineContractError(
                f"{key}.area_allocation_fraction must be in (0, 1]"
            )
        if not 0.0 <= coverage <= 1.0:
            raise RefineContractError(
                f"{key}.min_anchor_coverage_fraction must be in [0, 1]"
            )
        if not 0.0 <= off_anchor <= 0.25:
            raise RefineContractError(
                f"{key}.max_off_anchor_contact_fraction must be in [0, 0.25]"
            )
        if not 0.0 <= tolerance <= 0.10:
            raise RefineContractError(
                f"{key}.allocation_tolerance_fraction must be in [0, 0.10]"
            )
        return cls(
            anchor_segment_ids=anchors,
            area_allocation_fraction=allocation,
            depth_profile=DepthProfile.from_mapping(
                payload.get("depth_profile"), key=f"{key}.depth_profile"
            ),
            min_anchor_coverage_fraction=coverage,
            max_off_anchor_contact_fraction=off_anchor,
            allocation_tolerance_fraction=tolerance,
        )


@dataclass(frozen=True)
class PlannedInterface:
    interface_id: str
    source_component_id: str
    target_component_id: str
    anchor_segment: str
    allowed_edit_band_px: tuple[float, float]
    execution_contract: InterfaceExecutionContract
    prohibited_region_ids: tuple[str, ...]
    supporting_rule_ids: tuple[str, ...]
    expected_morphology: str
    confidence: float


@dataclass(frozen=True)
class ToolProgram:
    allowed_tools: tuple[str, ...]
    parameter_ranges: dict[str, Any]
    candidate_count: int = 12


@dataclass(frozen=True)
class EditPlan:
    schema_version: str
    case_id: str
    normalized_intent: str
    primitive_id: str
    source_labels: tuple[str, ...]
    target_label: str
    area_budget: AreaBudget
    candidate_interfaces: tuple[PlannedInterface, ...]
    tool_program: ToolProgram
    hard_invariants: tuple[str, ...]
    uncertainties: tuple[str, ...]
    planner_confidence: float
    escalation_reason: str | None = None
    resolved_area: ResolvedAreaContract | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EditPlan:
        if not isinstance(payload, Mapping):
            raise RefineContractError("EditPlan response must be a mapping")
        raw_interfaces = payload.get("candidate_interfaces")
        if not isinstance(raw_interfaces, list) or not raw_interfaces:
            raise RefineContractError("EditPlan requires at least one candidate interface")
        interfaces: list[PlannedInterface] = []
        for index, raw in enumerate(raw_interfaces):
            if not isinstance(raw, Mapping):
                raise RefineContractError(f"candidate_interfaces[{index}] must be a mapping")
            band = raw.get("allowed_edit_band_px")
            if (
                not isinstance(band, Sequence)
                or isinstance(band, (str, bytes))
                or len(band) != 2
            ):
                raise RefineContractError(
                    f"candidate_interfaces[{index}].allowed_edit_band_px must have two values"
                )
            confidence = _required_number(raw, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise RefineContractError("interface confidence must be in [0, 1]")
            interfaces.append(
                PlannedInterface(
                    interface_id=_required_string(raw, "interface_id"),
                    source_component_id=_required_string(raw, "source_component_id"),
                    target_component_id=_required_string(raw, "target_component_id"),
                    anchor_segment=_required_string(raw, "anchor_segment"),
                    allowed_edit_band_px=(float(band[0]), float(band[1])),
                    execution_contract=InterfaceExecutionContract.from_mapping(
                        raw.get("execution_contract"),
                        key=f"candidate_interfaces[{index}].execution_contract",
                    ),
                    prohibited_region_ids=_string_tuple(
                        raw.get("prohibited_region_ids", ()),
                        f"candidate_interfaces[{index}].prohibited_region_ids",
                    ),
                    supporting_rule_ids=_string_tuple(
                        raw.get("supporting_rule_ids", ()),
                        f"candidate_interfaces[{index}].supporting_rule_ids",
                    ),
                    expected_morphology=_required_string(raw, "expected_morphology"),
                    confidence=confidence,
                )
            )
        raw_program = payload.get("tool_program")
        if not isinstance(raw_program, Mapping):
            raise RefineContractError("EditPlan.tool_program must be a mapping")
        count = int(raw_program.get("candidate_count", 12))
        if not 1 <= count <= 48:
            raise RefineContractError("candidate_count must be in [1, 48]")
        planner_confidence = _required_number(payload, "planner_confidence")
        if not 0.0 <= planner_confidence <= 1.0:
            raise RefineContractError("planner_confidence must be in [0, 1]")
        escalation_reason = payload.get("escalation_reason")
        if escalation_reason is not None and not isinstance(escalation_reason, str):
            raise RefineContractError("escalation_reason must be a string or null")
        return cls(
            schema_version=_required_string(payload, "schema_version"),
            case_id=_required_string(payload, "case_id"),
            normalized_intent=_required_string(payload, "normalized_intent"),
            primitive_id=_required_string(payload, "primitive_id"),
            source_labels=_string_tuple(payload.get("source_labels"), "source_labels"),
            target_label=_required_string(payload, "target_label"),
            area_budget=AreaBudget.from_value(payload.get("area_budget")),
            candidate_interfaces=tuple(interfaces),
            tool_program=ToolProgram(
                allowed_tools=_string_tuple(raw_program.get("allowed_tools"), "allowed_tools"),
                parameter_ranges=dict(raw_program.get("parameter_ranges", {})),
                candidate_count=count,
            ),
            hard_invariants=_string_tuple(payload.get("hard_invariants", ()), "hard_invariants"),
            uncertainties=_string_tuple(payload.get("uncertainties", ()), "uncertainties"),
            planner_confidence=planner_confidence,
            escalation_reason=escalation_reason,
            resolved_area=ResolvedAreaContract.from_mapping(
                payload.get("resolved_area")
            ),
        )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateMask:
    candidate_id: str
    interface_id: str
    tool_name: str
    target_mask: np.ndarray
    change_region: np.ndarray
    tool_trace: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "interface_id": self.interface_id,
            "tool_name": self.tool_name,
            "changed_pixels": int(np.count_nonzero(self.change_region)),
            "tool_trace": dict(self.tool_trace),
        }


@dataclass(frozen=True)
class GateCheck:
    check_id: str
    passed: bool
    severity: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateReport:
    candidate_id: str
    passed: bool
    checks: tuple[GateCheck, ...]

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriticRanking:
    candidate_id: str
    score: float
    confidence: float
    supporting_rule_ids: tuple[str, ...]
    veto_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriticResult:
    rankings: tuple[CriticRanking, ...]
    abstain: bool
    summary: str
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowResult:
    status: str
    case_context: CaseContext
    scene_graph: SceneGraph | None
    edit_plan: EditPlan | None
    gate_reports: tuple[GateReport, ...]
    critic_result: CriticResult | None
    selected_candidate_id: str | None
    target_mask: np.ndarray | None
    abstain_reasons: tuple[str, ...]
    artifact_paths: dict[str, str]
    usage: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "case_context": self.case_context.to_metadata(),
            "scene_graph": self.scene_graph.to_metadata() if self.scene_graph else None,
            "edit_plan": self.edit_plan.to_metadata() if self.edit_plan else None,
            "gate_reports": [report.to_metadata() for report in self.gate_reports],
            "critic_result": asdict(self.critic_result) if self.critic_result else None,
            "selected_candidate_id": self.selected_candidate_id,
            "abstain_reasons": list(self.abstain_reasons),
            "artifact_paths": dict(self.artifact_paths),
            "usage": dict(self.usage),
        }


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RefineContractError(f"{key} is required and must be a non-empty string")
    return value.strip()


def _required_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise RefineContractError(f"{key} is required and must be numeric")
    return float(value)


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        raise RefineContractError(f"{key} must be a string or list of strings")
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise RefineContractError(f"{key} must contain non-empty strings")
    return tuple(item.strip() for item in values)

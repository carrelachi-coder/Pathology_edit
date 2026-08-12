"""Load and compose domain mechanisms with independent annotation/cell axes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from phase3_joint_edit_refine.models import JointCaseContext, JointContractError
from phase3_mask_edit_refine.skills import SkillRepository as MaskSkillRepository

from .evidence import EvidenceGovernance, SkillEvidenceStatus
from .execution_aliases import tissue_tool_primitive_id
from .schema import JointMechanismSkill, JointPrimitiveSkill, JointProfileContract


@dataclass(frozen=True)
class CellObservationProfile:
    profile_id: str
    version: str
    class_ids: tuple[int, ...]
    class_names: dict[int, str]
    tissue_compatible_classes: dict[str, tuple[int, ...]]
    required_checker_ids: tuple[str, ...]


@dataclass(frozen=True)
class CellPopulationProfile:
    profile_id: str
    pathology_domain_id: str
    version: str
    source_first: bool
    allow_cross_domain_fallback: bool
    allowed_cell_classes: tuple[int, ...]
    probnet_cancer_id: int
    probnet_dataset_name: str


@dataclass(frozen=True)
class JointSkillBundle:
    primitive: JointPrimitiveSkill
    mechanism: JointMechanismSkill
    annotation_profile: JointProfileContract
    cell_observation_profile: CellObservationProfile
    cell_population_profile: CellPopulationProfile
    support_status: str
    required_checker_ids: tuple[str, ...]
    active_rule_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence_status: dict[str, dict] = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "primitive_id": self.primitive.primitive_id,
            "primitive_scope": self.primitive.scope,
            "primitive_tissue_geometry_mode": (
                self.primitive.tissue_geometry_mode
            ),
            "primitive_tissue_topology_contract": {
                "allow_source_component_resolution": (
                    self.primitive.allow_source_component_resolution
                ),
                "allow_target_hole_resolution": (
                    self.primitive.allow_target_hole_resolution
                ),
                "maximum_source_component_changed_fraction": (
                    self.primitive.maximum_source_component_changed_fraction
                ),
                "minimum_source_component_remaining_px": (
                    self.primitive.minimum_source_component_remaining_px
                ),
            },
            "primitive_required_source_clearance_classes": list(
                self.primitive.required_source_clearance_classes
            ),
            "primitive_minimum_source_clearance_instances": (
                self.primitive.minimum_source_clearance_instances
            ),
            "mechanism_id": self.mechanism.mechanism_id,
            "required_auxiliary_structures": list(
                self.mechanism.representability.required_auxiliary_structures
            ),
            "receiving_auxiliary_structures": list(
                self.mechanism.representability.receiving_auxiliary_structures
            ),
            "protected_auxiliary_structures": list(
                self.mechanism.representability.protected_auxiliary_structures
            ),
            "compiled_layout_program": self.mechanism.cell_program.layout_for(
                self.primitive.primitive_id
            ),
            "mechanism_topology_fallback_by_primitive": {
                primitive_id: {
                    "geometry_mode": contract.geometry_mode,
                    "allow_source_component_resolution": (
                        contract.allow_source_component_resolution
                    ),
                    "allow_target_hole_resolution": (
                        contract.allow_target_hole_resolution
                    ),
                    "maximum_source_component_changed_fraction": (
                        contract.maximum_source_component_changed_fraction
                    ),
                    "minimum_source_component_remaining_px": (
                        contract.minimum_source_component_remaining_px
                    ),
                }
                for primitive_id, contract in sorted(
                    self.mechanism.tissue_program.topology_fallback_by_primitive.items()
                )
            },
            "annotation_profile_id": self.annotation_profile.annotation_profile_id,
            "annotation_supports_explicit_stroma": (
                self.annotation_profile.supports_explicit_stroma
            ),
            "cell_observation_profile_id": self.cell_observation_profile.profile_id,
            "cell_population_profile_id": self.cell_population_profile.profile_id,
            "support_status": self.support_status,
            "required_checker_ids": list(self.required_checker_ids),
            "active_rule_ids": list(self.active_rule_ids),
            "warnings": list(self.warnings),
            "evidence_status": dict(self.evidence_status),
            "source_paths": {
                "mechanism": self.mechanism.source_path,
                "primitive": self.primitive.source_path,
                "annotation_profile": self.annotation_profile.source_path,
            },
        }


class JointSkillRepository:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).parent / "catalog"
        self.mask_skills = MaskSkillRepository()
        self.primitives = self._load_primitives()
        self.mechanisms = self._load_mechanisms()
        self.annotation_profiles = self._load_annotation_profiles()
        self.cell_observation_profiles = self._load_cell_observation_profiles()
        self.cell_population_profiles = self._load_cell_population_profiles()
        self.execution_scope = self._load_execution_scope()
        self.evidence_governance = EvidenceGovernance(
            self.root / "evidence-governance-v2.json"
        )
        self.evidence_governance.validate_catalog_coverage(
            mechanism_ids=set(self.mechanisms),
            annotation_profile_ids=set(self.annotation_profiles),
        )
        self.skill_evidence_status: dict[str, SkillEvidenceStatus] = {}
        self.mechanism_resource_status = self._validate_skill_packages()
        self._validate_execution_scope()

    def _validate_skill_packages(self) -> dict[str, dict]:
        """Validate progressive-disclosure packages and calibration state.

        Runtime JSON is executable authority, while SKILL.md routes Planner
        retrieval and the evidence/statistics files disclose why a mechanism
        is or is not production eligible.  Missing resources fail startup;
        draft or uncalibrated resources remain usable only in research mode.
        """

        status: dict[str, dict] = {}
        specifications = (
            ("joint-mechanism", ("joint_contract.json", "evidence.json", "counterexamples.json", "statistics.json")),
            ("edit-primitive", ("primitive_contract.json", "evidence.json")),
            ("annotation-profile", ("joint_contract.json", "evidence.json")),
        )
        for kind, references in specifications:
            for directory in sorted((self.root / kind).glob("*")):
                if not directory.is_dir():
                    continue
                required_paths = [
                    directory / "SKILL.md",
                    directory / "agents" / "openai.yaml",
                    *[directory / "references" / name for name in references],
                ]
                missing = [str(path) for path in required_paths if not path.is_file()]
                if missing:
                    raise JointContractError(
                        f"incomplete {kind} skill package {directory.name}: "
                        + ", ".join(missing)
                    )
                if kind == "joint-mechanism":
                    statistics = _read_json(directory / "references" / "statistics.json")
                    evidence = _read_json(directory / "references" / "evidence.json")
                    contract = _read_json(
                        directory / "references" / "joint_contract.json"
                    )
                    evidence_status = self.evidence_governance.audit_mechanism(
                        contract
                    )
                    self.evidence_governance.validate_local_resource(
                        evidence, expected=evidence_status
                    )
                    self.skill_evidence_status[
                        f"joint-mechanism:{directory.name}"
                    ] = evidence_status
                    status[directory.name] = {
                        "statistics_status": str(statistics.get("status", "unknown")),
                        "production_allowed": bool(
                            statistics.get("production_allowed", False)
                        ) and evidence_status.production_allowed,
                        "dataset_statistics_status": str(
                            (evidence.get("dataset_statistics") or {}).get(
                                "status", "unknown"
                            )
                        ),
                        "evidence": evidence_status.to_metadata(),
                    }
                elif kind == "edit-primitive":
                    contract = _read_json(
                        directory / "references" / "primitive_contract.json"
                    )
                    evidence_status = self.evidence_governance.audit_primitive(
                        contract
                    )
                    self.evidence_governance.validate_local_resource(
                        _read_json(directory / "references" / "evidence.json"),
                        expected=evidence_status,
                    )
                    self.skill_evidence_status[
                        f"edit-primitive:{directory.name}"
                    ] = evidence_status
                elif kind == "annotation-profile":
                    contract = _read_json(
                        directory / "references" / "joint_contract.json"
                    )
                    evidence_status = self.evidence_governance.audit_profile(
                        contract
                    )
                    self.evidence_governance.validate_local_resource(
                        _read_json(directory / "references" / "evidence.json"),
                        expected=evidence_status,
                    )
                    self.skill_evidence_status[
                        f"annotation-profile:{directory.name}"
                    ] = evidence_status
        return status

    def _load_execution_scope(self) -> dict:
        payload = _read_json(self.root / "execution-scope-v1.json")
        if payload.get("schema_version") != "joint-execution-scope-v1":
            raise JointContractError("unsupported joint execution scope schema")
        if payload.get("policy") != "fail_closed":
            raise JointContractError("joint primitive scope must fail closed")
        return payload

    def _validate_execution_scope(self) -> None:
        executable = set(self.execution_scope.get("executable_primitives", []))
        closed = set(self.execution_scope.get("closed_primitives", {}))
        catalog = set(self.primitives)
        if executable & closed or executable | closed != catalog:
            raise JointContractError(
                "joint execution scope must classify every catalog primitive once"
            )
        covered = {
            primitive_id
            for item in self.mechanisms.values()
            for primitive_id in item.supported_primitives
        }
        if executable - covered:
            raise JointContractError(
                "executable joint primitives have no mechanism coverage: "
                + ", ".join(sorted(executable - covered))
            )
        for mechanism in self.mechanisms.values():
            burden_directions = {
                primitive_id
                for primitive_id in mechanism.supported_primitives
                if primitive_id
                in {
                    "tumor-burden-increase-v1",
                    "tumor-burden-decrease-v1",
                }
            }
            if len(burden_directions) > 1:
                raise JointContractError(
                    f"{mechanism.mechanism_id} conflates tumor growth with "
                    "unspecified regression; burden directions require separate "
                    "mechanism contracts"
                )

    @property
    def executable_primitive_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.execution_scope["executable_primitives"]))

    def primitive_scope_reason(self, primitive_id: str) -> str | None:
        closed = self.execution_scope.get("closed_primitives", {})
        legacy = self.execution_scope.get("legacy_primitives_not_yet_joint", {})
        if primitive_id in closed:
            return str(closed[primitive_id])
        if primitive_id in legacy:
            return str(legacy[primitive_id])
        return None

    def _load_mechanisms(self) -> dict[str, JointMechanismSkill]:
        result: dict[str, JointMechanismSkill] = {}
        for path in sorted(self.root.glob("joint-mechanism/*/references/joint_contract.json")):
            payload = _read_json(path)
            item = JointMechanismSkill.from_mapping(payload, source_path=str(path))
            if item.mechanism_id in result:
                raise JointContractError(f"duplicate joint mechanism: {item.mechanism_id}")
            result[item.mechanism_id] = item
        if not result:
            raise JointContractError(f"no joint mechanisms found below {self.root}")
        return result

    def _load_primitives(self) -> dict[str, JointPrimitiveSkill]:
        result: dict[str, JointPrimitiveSkill] = {}
        for path in sorted(
            self.root.glob("edit-primitive/*/references/primitive_contract.json")
        ):
            payload = _read_json(path)
            item = JointPrimitiveSkill.from_mapping(payload, source_path=str(path))
            if item.primitive_id in result:
                raise JointContractError(
                    f"duplicate joint primitive: {item.primitive_id}"
                )
            result[item.primitive_id] = item
        if not result:
            raise JointContractError(f"no joint primitives found below {self.root}")
        return result

    def _load_annotation_profiles(self) -> dict[str, JointProfileContract]:
        result = {}
        for path in sorted(self.root.glob("annotation-profile/*/references/joint_contract.json")):
            payload = _read_json(path)
            item = JointProfileContract.from_mapping(payload, source_path=str(path))
            result[item.annotation_profile_id] = item
        return result

    def _load_cell_observation_profiles(self) -> dict[str, CellObservationProfile]:
        path = self.root / "cell-observation-profile" / "profiles.json"
        payload = _read_json(path)
        result = {}
        for raw in payload.get("profiles", []):
            profile_id = str(raw["profile_id"])
            result[profile_id] = CellObservationProfile(
                profile_id=profile_id,
                version=str(raw["version"]),
                class_ids=tuple(int(value) for value in raw["class_ids"]),
                class_names={int(key): str(value) for key, value in raw["class_names"].items()},
                tissue_compatible_classes={
                    str(label): tuple(int(value) for value in values)
                    for label, values in raw["tissue_compatible_classes"].items()
                },
                required_checker_ids=tuple(str(value) for value in raw["required_checker_ids"]),
            )
        return result

    def _load_cell_population_profiles(self) -> dict[str, CellPopulationProfile]:
        path = self.root / "cell-population-profile" / "profiles.json"
        payload = _read_json(path)
        result = {}
        for raw in payload.get("profiles", []):
            profile_id = str(raw["profile_id"])
            result[profile_id] = CellPopulationProfile(
                profile_id=profile_id,
                pathology_domain_id=str(raw["pathology_domain_id"]),
                version=str(raw["version"]),
                source_first=bool(raw["source_first"]),
                allow_cross_domain_fallback=bool(raw["allow_cross_domain_fallback"]),
                allowed_cell_classes=tuple(int(value) for value in raw["allowed_cell_classes"]),
                probnet_cancer_id=int(raw["probnet_cancer_id"]),
                probnet_dataset_name=str(raw["probnet_dataset_name"]),
            )
        return result

    def mechanisms_for(self, *, pathology_domain_id: str, primitive_id: str) -> tuple[JointMechanismSkill, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self.mechanisms.values()
                    if item.pathology_domain_id == pathology_domain_id
                    and primitive_id in item.supported_primitives
                ),
                key=lambda item: item.mechanism_id,
            )
        )

    def validate_checker_registry(
        self, available_checker_ids: tuple[str, ...] | set[str]
    ) -> None:
        """Fail process startup if any catalog hard checker is unavailable."""

        required = set()
        for mechanism in self.mechanisms.values():
            required.update(mechanism.tissue_program.required_checker_ids)
            required.update(mechanism.cell_program.required_checker_ids)
            required.update(mechanism.joint_gate_ids)
            required.add(
                f"mechanism_postcondition:{mechanism.mechanism_id}"
            )
        for primitive in self.primitives.values():
            required.update(primitive.required_checker_ids)
        for profile in self.annotation_profiles.values():
            required.update(profile.required_checker_ids)
        for profile in self.cell_observation_profiles.values():
            required.update(profile.required_checker_ids)
        missing = sorted(required - set(available_checker_ids))
        if missing:
            raise JointContractError(
                "joint catalog references unavailable hard checkers: "
                + ", ".join(missing)
            )

    def eligible_mechanisms_for_case(
        self,
        *,
        case: JointCaseContext,
        available_checker_ids: tuple[str, ...] | set[str],
        production: bool,
    ) -> tuple[tuple[JointMechanismSkill, ...], dict[str, str]]:
        """Apply all four skill axes before exposing choices to a Planner."""

        if case.primitive_id not in self.execution_scope["executable_primitives"]:
            reason = self.primitive_scope_reason(case.primitive_id)
            return (), {
                "execution_scope": reason
                or "primitive is not in the reviewed joint executable scope"
            }

        eligible = []
        rejected = {}
        for mechanism in self.mechanisms_for(
            pathology_domain_id=case.pathology_domain_id,
            primitive_id=case.primitive_id,
        ):
            try:
                self.compose(
                    case=case,
                    mechanism_id=mechanism.mechanism_id,
                    available_checker_ids=available_checker_ids,
                    production=production,
                )
            except JointContractError as exc:
                rejected[mechanism.mechanism_id] = str(exc)
            else:
                eligible.append(mechanism)
        return tuple(eligible), rejected

    def compose(
        self,
        *,
        case: JointCaseContext,
        mechanism_id: str,
        available_checker_ids: tuple[str, ...] | set[str],
        production: bool,
    ) -> JointSkillBundle:
        if case.primitive_id not in self.execution_scope["executable_primitives"]:
            reason = self.primitive_scope_reason(case.primitive_id)
            raise JointContractError(
                "joint primitive is explicitly closed: "
                + (
                    reason
                    or "primitive is outside the reviewed executable scope"
                )
            )
        try:
            mechanism = self.mechanisms[mechanism_id]
            primitive_contract = self.primitives[case.primitive_id]
            annotation = self.annotation_profiles[case.annotation_profile_id]
            observation = self.cell_observation_profiles[case.cell_observation_profile_id]
            population = self.cell_population_profiles[case.cell_population_profile_id]
        except KeyError as exc:
            raise JointContractError(f"unknown joint skill/profile: {exc.args[0]}") from exc
        if mechanism.pathology_domain_id != case.pathology_domain_id:
            raise JointContractError("joint mechanism does not belong to pathology domain")
        if case.primitive_id not in mechanism.supported_primitives:
            raise JointContractError("joint mechanism does not support requested primitive")
        if (
            case.primitive_id == "stroma-increase-v1"
            and not annotation.supports_explicit_stroma
        ):
            raise JointContractError(
                "annotation profile has no explicit stromal authority; a non-gland "
                "complement or Other tissue label cannot authorize stroma increase"
            )
        if case.primitive_id == "stroma-increase-v1":
            if "treatment-associated" not in mechanism_id:
                raise JointContractError(
                    "Tumor-to-Stroma replacement is not owned by a growth mechanism"
                )
            if case.semantic_intent.get("treatment_context") != "post_treatment":
                raise JointContractError(
                    "treatment-associated stromal replacement requires an explicit "
                    "post-treatment context from the Semantic Parser; H&E cannot "
                    "invent treatment history"
                )
        if primitive_contract.scope == "tissue_and_cell" and case.joint_area_budget is None:
            raise JointContractError("tissue primitive requires joint_area_budget")
        if primitive_contract.scope == "cell_only" and case.cell_count_extent_budget is None:
            raise JointContractError("cell-only primitive requires cell_count_extent_budget")
        if primitive_contract.scope == "cell_only":
            budget = case.cell_count_extent_budget
            assert budget is not None
            if (
                budget.min_delta_count
                < primitive_contract.minimum_effect_delta_count
                or budget.max_delta_count
                < primitive_contract.minimum_effect_delta_count
            ):
                raise JointContractError(
                    "cell-only budget is below the skill-owned minimum effect count"
                )
        schema = self.annotation_schema(case.annotation_profile_id)
        if primitive_contract.tissue_action == "required":
            tool_primitive_id = tissue_tool_primitive_id(case.primitive_id)
            primitive = self.mask_skills.get(
                tool_primitive_id, expected_kind="edit_primitive"
            )
            target = primitive.capabilities.get("target_label")
            target_options = primitive.capabilities.get("target_label_options", [])
            if target not in schema.readable_labels and isinstance(target_options, list):
                target = next(
                    (item for item in target_options if item in schema.readable_labels),
                    target,
                )
            source_options = primitive.capabilities.get("source_label_options", [])
            if target not in schema.readable_labels or not any(
                item in schema.readable_labels for item in source_options
            ):
                raise JointContractError(
                    "annotation profile cannot represent the primitive source/target semantics"
                )
        elif not set(primitive_contract.host_tissue_labels).intersection(
            schema.readable_labels
        ):
            raise JointContractError(
                "annotation profile cannot represent a legal host tissue for the cell-only primitive"
            )
        if population.pathology_domain_id != case.pathology_domain_id:
            raise JointContractError(
                "cell population profile pathology domain mismatch; cross-domain fallback is forbidden"
            )
        if population.allow_cross_domain_fallback:
            raise JointContractError("joint v1 forbids cross-domain population fallback")
        if set(mechanism.representability.required_cell_classes) - set(observation.class_ids):
            raise JointContractError("cell observation profile cannot represent mechanism")
        if set(mechanism.cell_program.allowed_cell_classes) - set(population.allowed_cell_classes):
            raise JointContractError("cell population profile cannot realize mechanism classes")
        if mechanism_id in annotation.unavailable_mechanisms:
            support_status = "unsupported"
        elif mechanism_id in annotation.conditional_mechanisms:
            support_status = "conditionally_supported"
        else:
            support_status = mechanism.representability.status
        if support_status in {"unsupported", "render_only"}:
            raise JointContractError(
                f"mechanism {mechanism_id} is {support_status} for {case.annotation_profile_id}"
            )
        if production and (
            primitive_contract.review_status != "internally_reviewed"
            or mechanism.review_status != "internally_reviewed"
            or annotation.review_status != "internally_reviewed"
        ):
            raise JointContractError("production joint execution requires internally reviewed skills")
        evidence_statuses = {
            "primitive": self.skill_evidence_status[
                f"edit-primitive:{case.primitive_id}"
            ],
            "mechanism": self.skill_evidence_status[
                f"joint-mechanism:{mechanism_id}"
            ],
            "annotation_profile": self.skill_evidence_status[
                f"annotation-profile:{case.annotation_profile_id}"
            ],
        }
        if production and not all(
            item.production_allowed for item in evidence_statuses.values()
        ):
            gaps = []
            for axis, item in evidence_statuses.items():
                gaps.extend(f"{axis}: {gap}" for gap in item.gaps)
            raise JointContractError(
                "production joint execution has unresolved evidence gaps: "
                + "; ".join(gaps)
            )
        resource_status = self.mechanism_resource_status.get(mechanism_id, {})
        if production and (
            resource_status.get("statistics_status") != "calibrated"
            or resource_status.get("production_allowed") is not True
        ):
            raise JointContractError(
                "production joint execution requires calibrated, explicitly "
                "production-authorized mechanism statistics"
            )
        required = set(mechanism.joint_gate_ids)
        required.update(primitive_contract.required_checker_ids)
        required.update(mechanism.tissue_program.required_checker_ids)
        required.update(mechanism.cell_program.required_checker_ids)
        required.update(annotation.required_checker_ids)
        required.update(observation.required_checker_ids)
        missing = sorted(required - set(available_checker_ids))
        if missing:
            raise JointContractError(
                "joint skills reference unavailable checkers: " + ", ".join(missing)
            )
        warnings = []
        if support_status == "conditionally_supported":
            warnings.append("mechanism requires Planner evidence and representability checks")
        return JointSkillBundle(
            primitive=primitive_contract,
            mechanism=mechanism,
            annotation_profile=annotation,
            cell_observation_profile=observation,
            cell_population_profile=population,
            support_status=support_status,
            required_checker_ids=tuple(sorted(required)),
            active_rule_ids=tuple(
                dict.fromkeys(
                    [
                        *primitive_contract.required_checker_ids,
                        *mechanism.tissue_program.required_checker_ids,
                        *mechanism.cell_program.required_checker_ids,
                        *mechanism.joint_gate_ids,
                        *mechanism.coupling.compatibility_rule_ids,
                        *annotation.required_checker_ids,
                        *observation.required_checker_ids,
                    ]
                )
            ),
            warnings=tuple(warnings),
            evidence_status={
                key: value.to_metadata()
                for key, value in evidence_statuses.items()
            },
        )

    def annotation_schema(self, annotation_profile_id: str):
        return self.mask_skills.annotation_schema(annotation_profile_id)

    def capability_matrix(self) -> tuple[dict, ...]:
        """Return auditable domain×profile×mechanism support before case observations."""

        rows = []
        for mechanism in sorted(self.mechanisms.values(), key=lambda item: item.mechanism_id):
            population = next(
                item
                for item in self.cell_population_profiles.values()
                if item.pathology_domain_id == mechanism.pathology_domain_id
            )
            for annotation in sorted(self.annotation_profiles.values(), key=lambda item: item.annotation_profile_id):
                supported_primitives = []
                semantic_failures = []
                for primitive_id in mechanism.supported_primitives:
                    try:
                        schema = self.annotation_schema(annotation.annotation_profile_id)
                        primitive_contract = self.primitives[primitive_id]
                        if (
                            primitive_id == "stroma-increase-v1"
                            and not annotation.supports_explicit_stroma
                        ):
                            raise JointContractError(
                                "profile has no explicit stromal authority"
                            )
                        if primitive_contract.tissue_action == "required":
                            tool_primitive_id = tissue_tool_primitive_id(
                                primitive_id
                            )
                            primitive = self.mask_skills.get(
                                tool_primitive_id,
                                expected_kind="edit_primitive",
                            )
                            target = primitive.capabilities.get("target_label")
                            options = primitive.capabilities.get(
                                "target_label_options", []
                            )
                            if target not in schema.readable_labels and isinstance(options, list):
                                target = next(
                                    (item for item in options if item in schema.readable_labels),
                                    target,
                                )
                            sources = primitive.capabilities.get(
                                "source_label_options", []
                            )
                            if target not in schema.readable_labels or not any(
                                item in schema.readable_labels for item in sources
                            ):
                                raise JointContractError("semantic labels unavailable")
                        elif not set(
                            primitive_contract.host_tissue_labels
                        ).intersection(schema.readable_labels):
                            raise JointContractError("cell host labels unavailable")
                        supported_primitives.append(primitive_id)
                    except (JointContractError, ValueError) as exc:
                        semantic_failures.append({"primitive_id": primitive_id, "reason": str(exc)})
                if not supported_primitives or mechanism.mechanism_id in annotation.unavailable_mechanisms:
                    status = "unsupported"
                elif (
                    mechanism.mechanism_id in annotation.conditional_mechanisms
                    or mechanism.representability.status == "conditionally_supported"
                    or mechanism.representability.required_auxiliary_structures
                ):
                    status = "conditionally_supported"
                else:
                    status = mechanism.representability.status
                rows.append(
                    {
                        "pathology_domain_id": mechanism.pathology_domain_id,
                        "annotation_profile_id": annotation.annotation_profile_id,
                        "mechanism_id": mechanism.mechanism_id,
                        "cell_observation_profile_id": next(iter(self.cell_observation_profiles)),
                        "cell_population_profile_id": population.profile_id,
                        "status": status,
                        "supported_primitives": supported_primitives,
                        "semantic_failures": semantic_failures,
                        "required_auxiliary_structures": list(mechanism.representability.required_auxiliary_structures),
                        "review_status": mechanism.review_status,
                        "resource_status": self.mechanism_resource_status.get(
                            mechanism.mechanism_id, {}
                        ),
                    }
                )
        return tuple(rows)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointContractError(f"could not load joint skill resource {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JointContractError(f"joint skill resource root must be an object: {path}")
    return payload

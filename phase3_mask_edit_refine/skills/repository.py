"""Discovery, validation, and dual-axis composition for runtime skills."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import RefineContractError

from .schema import (
    ActiveKnowledgeBundle,
    ResolvedEditContract,
    SkillPackage,
)


class SkillRepository:
    """Load application-owned skill packages without installing global skills."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).parent / "catalog"
        self._skills = self._load_all()

    def _load_all(self) -> dict[str, SkillPackage]:
        packages: dict[str, SkillPackage] = {}
        for path in sorted(self.root.glob("*/*/references/rules.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RefineContractError(f"could not load skill package {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise RefineContractError(f"skill package root must be an object: {path}")
            contract_path = path.with_name("mask_contract.json")
            if contract_path.is_file():
                try:
                    contract_payload = json.loads(
                        contract_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise RefineContractError(
                        f"could not load mask contract {contract_path}: {exc}"
                    ) from exc
                if not isinstance(contract_payload, dict):
                    raise RefineContractError(
                        f"mask contract root must be an object: {contract_path}"
                    )
                if contract_payload.get("skill_id") != payload.get("skill_id"):
                    raise RefineContractError(
                        f"mask contract skill mismatch: {contract_path}"
                    )
                constraints = contract_payload.get("constraints")
                if not isinstance(constraints, list):
                    raise RefineContractError(
                        f"mask contract constraints must be a list: {contract_path}"
                    )
                payload = dict(payload)
                payload["mask_constraints"] = constraints
            package = SkillPackage.from_mapping(payload, source_path=str(path))
            _validate_non_breast_execution_authority(package)
            if package.skill_id in packages:
                raise RefineContractError(f"duplicate skill_id: {package.skill_id}")
            packages[package.skill_id] = package
        if not packages:
            raise RefineContractError(f"no skill packages found below {self.root}")
        return packages

    def get(self, skill_id: str, *, expected_kind: str | None = None) -> SkillPackage:
        try:
            package = self._skills[skill_id]
        except KeyError as exc:
            raise RefineContractError(f"unknown skill_id: {skill_id}") from exc
        if expected_kind and package.skill_kind != expected_kind:
            raise RefineContractError(
                f"skill {skill_id} has kind {package.skill_kind}, expected {expected_kind}"
            )
        return package

    def list(self, *, kind: str | None = None) -> tuple[str, ...]:
        return tuple(
            sorted(
                skill_id
                for skill_id, package in self._skills.items()
                if kind is None or package.skill_kind == kind
            )
        )

    def annotation_schema(self, annotation_profile_id: str) -> MaskProfileSchema:
        package = self.get(annotation_profile_id, expected_kind="annotation_profile")
        dataset_config_name = package.capabilities.get("dataset_config_name")
        if not isinstance(dataset_config_name, str) or not dataset_config_name:
            raise RefineContractError(
                f"annotation profile {annotation_profile_id} has no dataset_config_name"
            )
        schema = MaskProfileSchema.from_reference_profile(dataset_config_name)
        raw_partitions = package.capabilities.get(
            "component_partition_fine_ids", {}
        )
        if raw_partitions:
            if not isinstance(raw_partitions, dict):
                raise RefineContractError(
                    "component_partition_fine_ids must be a mapping"
                )
            partitions: dict[str, tuple[tuple[int, ...], ...]] = {}
            for label, groups in raw_partitions.items():
                if label not in schema.readable_labels or not isinstance(groups, list):
                    raise RefineContractError(
                        f"invalid component partition for {label!r}"
                    )
                normalized = tuple(
                    tuple(int(value) for value in group)
                    for group in groups
                    if isinstance(group, list) and group
                )
                available = set(schema.resolve_fine_ids(str(label)))
                flattened = {value for group in normalized for value in group}
                if not normalized or not flattened.issubset(available):
                    raise RefineContractError(
                        f"component partition for {label!r} contains unavailable fine IDs"
                    )
                partitions[str(label)] = normalized
            schema = replace(
                schema,
                component_partition_fine_ids=partitions,
            )
        return schema

    def compose(
        self,
        *,
        pathology_domain_id: str,
        annotation_profile_id: str,
        primitive_id: str,
        production: bool,
        available_checker_ids: Iterable[str] = (),
        case_provenance: dict[str, object] | None = None,
        available_auxiliary_authority_digests: Mapping[str, str] | None = None,
    ) -> ActiveKnowledgeBundle:
        pathology = self.get(pathology_domain_id, expected_kind="pathology_domain")
        annotation = self.get(annotation_profile_id, expected_kind="annotation_profile")
        primitive = self.get(primitive_id, expected_kind="edit_primitive")
        packages = (pathology, annotation, primitive)
        if production:
            uncertified = [
                package.skill_id
                for package in packages
                if package.review_status != "internally_reviewed"
            ]
            if uncertified:
                raise RefineContractError(
                    "production execution requires internally_reviewed skills: "
                    + ", ".join(uncertified)
                )
        annotation = self._attach_annotation_statistics(annotation, production=production)

        available_labels = _strings_from_capability(annotation, "canonical_labels")
        target_label = primitive.capabilities.get("target_label")
        target_options = primitive.capabilities.get("target_label_options", [])
        if target_label not in available_labels and isinstance(target_options, list):
            target_label = next(
                (item for item in target_options if isinstance(item, str) and item in available_labels),
                target_label,
            )
        source_options = _strings_from_capability(primitive, "source_label_options")
        allowed_tools = _strings_from_capability(primitive, "allowed_tools")
        if not isinstance(target_label, str) or target_label not in available_labels:
            raise RefineContractError(
                f"annotation profile {annotation_profile_id} cannot represent target label "
                f"{target_label!r} for {primitive_id}"
            )
        resolved_sources = tuple(label for label in source_options if label in available_labels)
        if not resolved_sources:
            raise RefineContractError(
                f"annotation profile {annotation_profile_id} has no legal source label for "
                f"{primitive_id}; configured options={list(source_options)}"
            )

        auxiliary_authority_digests = dict(
            available_auxiliary_authority_digests or {}
        )
        non_breast_execution = bool(
            pathology_domain_id != "breast-invasive-carcinoma-v1"
            or annotation_profile_id != "bcss-semantic-v1"
        )
        active_rules_list = []
        for package in packages:
            for rule in package.rules:
                if rule.scope.startswith("reader_only_"):
                    continue
                if not non_breast_execution:
                    active_rules_list.append(rule)
                    continue
                if not _rule_applies(
                    rule.applies_when,
                    pathology_domain_id=pathology_domain_id,
                    annotation_profile_id=annotation_profile_id,
                    primitive_id=primitive_id,
                    annotation=annotation,
                    relevant_labels={target_label, *resolved_sources},
                    case_provenance=case_provenance,
                ):
                    continue
                active_rules_list.append(rule)
        active_rules = tuple(active_rules_list)
        if non_breast_execution:
            _validate_composed_rule_authority(
                rules=active_rules,
                annotation=annotation,
                case_provenance=case_provenance,
                available_auxiliary_authority_digests=auxiliary_authority_digests,
                require_materialized_provenance=case_provenance is not None,
            )
        active_mask_constraints = tuple(
            constraint
            for package in packages
            for constraint in package.mask_constraints
            if _mask_constraint_applies(
                constraint.applies_when,
                pathology_domain_id=pathology_domain_id,
                annotation_profile_id=annotation_profile_id,
                primitive_id=primitive_id,
            )
        )
        required_check_ids = tuple(
            sorted(
                {
                    rule.deterministic_check_id
                    for rule in active_rules
                    if rule.severity == "hard" and rule.deterministic_check_id
                }.union(
                    checker_id
                    for constraint in active_mask_constraints
                    for checker_id in constraint.checker_ids
                )
            )
        )
        missing_checkers = sorted(set(required_check_ids) - set(available_checker_ids))
        if missing_checkers:
            raise RefineContractError(
                "hard skill rules reference unavailable checkers: "
                + ", ".join(missing_checkers)
            )

        warnings: list[str] = []
        if any(package.review_status != "internally_reviewed" for package in packages):
            warnings.append("research_only_unreviewed_skills")
        limitations = annotation.capabilities.get("semantic_limitations", [])
        if isinstance(limitations, list):
            warnings.extend(str(item) for item in limitations)
        return ActiveKnowledgeBundle(
            pathology_domain=pathology,
            annotation_profile=annotation,
            edit_primitive=primitive,
            edit_contract=ResolvedEditContract(
                primitive_id=primitive_id,
                source_label_options=resolved_sources,
                target_label=target_label,
                allowed_tools=allowed_tools,
                required_check_ids=required_check_ids,
            ),
            active_rules=active_rules,
            active_mask_constraints=active_mask_constraints,
            warnings=tuple(warnings),
        )

    def _attach_annotation_statistics(
        self, package: SkillPackage, *, production: bool
    ) -> SkillPackage:
        artifact = package.capabilities.get("statistics_artifact")
        if artifact is None:
            if production:
                raise RefineContractError(
                    f"annotation profile {package.skill_id} has no certified statistics artifact"
                )
            return package
        if not isinstance(artifact, str) or not artifact:
            raise RefineContractError(
                f"annotation profile {package.skill_id} statistics_artifact must be a path"
            )
        path = Path(package.source_path).parent / artifact
        try:
            statistics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RefineContractError(f"could not load statistics artifact {path}: {exc}") from exc
        if not isinstance(statistics, dict):
            raise RefineContractError(f"statistics artifact root must be an object: {path}")
        if statistics.get("annotation_profile_id") != package.skill_id:
            raise RefineContractError(
                f"statistics artifact profile mismatch for {package.skill_id}"
            )
        if production and statistics.get("review_status") != "internally_reviewed":
            raise RefineContractError(
                f"statistics artifact for {package.skill_id} is not internally_reviewed"
            )
        capabilities = dict(package.capabilities)
        capabilities["empirical_statistics"] = statistics
        capabilities["statistics_artifact_resolved"] = str(path)
        return replace(package, capabilities=capabilities)


def _strings_from_capability(package: SkillPackage, key: str) -> tuple[str, ...]:
    value = package.capabilities.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RefineContractError(
            f"skill {package.skill_id} capability {key} must be a list of strings"
        )
    return tuple(value)


_NON_BREAST_EXECUTION_SKILLS = frozenset(
    {
        "glas-gland-v1",
        "panda-gleason-v1",
        "ignite-semantic-v1",
        "puma-semantic-v1",
        "orca-semantic-v1",
        "colorectal-adenocarcinoma-v1",
        "prostate-adenocarcinoma-v1",
        "lung-carcinoma-v1",
        "melanoma-v1",
        "oral-squamous-cell-carcinoma-v1",
    }
)
_EXECUTION_HE_PATTERN = re.compile(
    r"(?:\bH\s*&\s*E\b|\bH&E\b|source[_ -]?he\b|raw histology)",
    flags=re.IGNORECASE,
)


def _validate_non_breast_execution_authority(package: SkillPackage) -> None:
    """Reject catalog drift unless execution authority is positively typed."""

    if package.skill_id not in _NON_BREAST_EXECUTION_SKILLS:
        return
    for rule in package.rules:
        if rule.scope.startswith("reader_only_"):
            if (
                rule.critic_requirement
                or rule.deterministic_check_id
                or rule.execution_role
                or rule.observation_authority
                or rule.selection_preference
            ):
                raise RefineContractError(
                    f"reader-only pathology fact {rule.rule_id} carries execution authority"
                )
            continue
        if rule.critic_requirement:
            raise RefineContractError(
                f"non-Breast rule {rule.rule_id} grants execution critic veto authority"
            )
        if not rule.execution_role or not rule.observation_authority:
            raise RefineContractError(
                f"non-Breast rule {rule.rule_id} lacks typed observation authority"
            )
    for constraint in package.mask_constraints:
        authority_text = " ".join(
            (
                constraint.mask_statement,
                *constraint.observability,
                *constraint.required_inputs,
            )
        )
        if constraint.critic_requirement:
            raise RefineContractError(
                f"non-Breast constraint {constraint.constraint_id} grants critic veto authority"
            )
        if "source_he" in constraint.observability or _EXECUTION_HE_PATTERN.search(
            authority_text
        ):
            raise RefineContractError(
                f"non-Breast constraint {constraint.constraint_id} grants execution H&E authority"
            )


def validate_active_bundle_authority(
    bundle: ActiveKnowledgeBundle,
    *,
    case_provenance: dict[str, object],
    available_auxiliary_authority_digests: Mapping[str, str] | None = None,
) -> None:
    """Revalidate a non-Breast bundle at every direct agent entry point."""

    non_breast_execution = bool(
        bundle.pathology_domain.skill_id != "breast-invasive-carcinoma-v1"
        or bundle.annotation_profile.skill_id != "bcss-semantic-v1"
    )
    if not non_breast_execution:
        return
    _validate_composed_rule_authority(
        rules=bundle.active_rules,
        annotation=bundle.annotation_profile,
        case_provenance=case_provenance,
        available_auxiliary_authority_digests=dict(
            available_auxiliary_authority_digests or {}
        ),
        require_materialized_provenance=True,
    )


def _validate_composed_rule_authority(
    *,
    rules: Iterable,
    annotation: SkillPackage,
    case_provenance: dict[str, object] | None,
    available_auxiliary_authority_digests: dict[str, str],
    require_materialized_provenance: bool,
) -> None:
    structure_bindings = annotation.capabilities.get(
        "profile_owned_structure_authorities", {}
    )
    if not isinstance(structure_bindings, dict):
        raise RefineContractError(
            "profile_owned_structure_authorities must be a mapping"
        )
    required_provenance = _strings_from_capability(
        annotation, "required_provenance_fields"
    )
    for rule in rules:
        if rule.scope.startswith("reader_only_"):
            raise RefineContractError(
                f"reader-only rule {rule.rule_id} entered the execution bundle"
            )
        if not rule.execution_role or not rule.observation_authority:
            raise RefineContractError(
                f"execution rule {rule.rule_id} lacks typed observation authority"
            )
        authority = {
            (item.source, item.binding) for item in rule.observation_authority
        }
        sources = {source for source, _binding in authority}
        required_sources = {
            "deterministic_mask_invariant": {
                "tissue_mask",
                "scene_graph",
                "deterministic_metric",
            },
            "provenance_precondition": {
                "case_provenance",
                "deterministic_metric",
            },
            "semantic_capability_precondition": {
                "instruction_semantic_intent",
                "tissue_mask",
                "deterministic_metric",
            },
            "profile_auxiliary_selection_preference": {
                "profile_owned_auxiliary_map",
                "candidate_certificate",
                "deterministic_metric",
            },
            "certified_candidate_selection_preference": {
                "candidate_certificate",
                "deterministic_metric",
            },
        }[rule.execution_role]
        allowed_sources = {
            "deterministic_mask_invariant": required_sources
            | {"nuclei_mask", "candidate_certificate"},
            "provenance_precondition": required_sources
            | {"instruction_semantic_intent"},
            "semantic_capability_precondition": required_sources
            | {"case_provenance"},
            "profile_auxiliary_selection_preference": required_sources
            | {"tissue_mask", "nuclei_mask", "scene_graph"},
            "certified_candidate_selection_preference": required_sources
            | {"tissue_mask", "nuclei_mask", "scene_graph"},
        }[rule.execution_role]
        missing_sources = sorted(required_sources - sources)
        unexpected_sources = sorted(sources - allowed_sources)
        if missing_sources or unexpected_sources:
            raise RefineContractError(
                f"rule {rule.rule_id} has authority sources detached from role "
                f"{rule.execution_role}: missing={missing_sources}, "
                f"unexpected={unexpected_sources}"
            )
        for source, binding in authority:
            if source == "instruction_semantic_intent" and binding != "primitive_id":
                raise RefineContractError(
                    f"rule {rule.rule_id} has an unbound semantic-intent authority"
                )
            if source == "tissue_mask" and binding != "source_mask_sha256":
                raise RefineContractError(
                    f"rule {rule.rule_id} has an unbound tissue-mask authority"
                )
            if source == "nuclei_mask" and binding != "source_nuclei_sha256":
                raise RefineContractError(
                    f"rule {rule.rule_id} has an unbound nuclei-mask authority"
                )
            if source == "scene_graph" and binding != "compiler_scene_graph":
                raise RefineContractError(
                    f"rule {rule.rule_id} has an unbound scene-graph authority"
                )
            if (
                source == "candidate_certificate"
                and binding != "compiler_candidate_certificate"
            ):
                raise RefineContractError(
                    f"rule {rule.rule_id} has an unbound candidate-certificate authority"
                )
            if source == "deterministic_metric":
                expected = (
                    f"checker:{rule.deterministic_check_id}"
                    if rule.deterministic_check_id
                    else None
                )
                if binding != expected:
                    raise RefineContractError(
                        f"rule {rule.rule_id} has a detached deterministic metric"
                    )
            if source == "case_provenance":
                if binding == "profile_required_provenance":
                    declared_keys = required_provenance
                elif binding.startswith("provenance:"):
                    declared_keys = (binding.removeprefix("provenance:"),)
                else:
                    raise RefineContractError(
                        f"rule {rule.rule_id} has an unbound provenance authority"
                    )
                if require_materialized_provenance:
                    missing = [
                        key
                        for key in declared_keys
                        if not isinstance((case_provenance or {}).get(key), str)
                        or not str((case_provenance or {}).get(key)).strip()
                    ]
                    if missing:
                        raise RefineContractError(
                            f"rule {rule.rule_id} lacks materialized provenance authority: "
                            + ", ".join(missing)
                        )
            if source == "profile_owned_auxiliary_map":
                prefix = "profile_auxiliary:"
                if not binding.startswith(prefix):
                    raise RefineContractError(
                        f"rule {rule.rule_id} has an unbound profile auxiliary authority"
                    )
                auxiliary_id = binding.removeprefix(prefix)
                digest = available_auxiliary_authority_digests.get(auxiliary_id)
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise RefineContractError(
                        f"rule {rule.rule_id} lacks digest-bound profile auxiliary authority: "
                        f"{auxiliary_id}"
                    )

        structures = _condition_values(rule.applies_when.get("structure"))
        if structures:
            if rule.execution_role != "profile_auxiliary_selection_preference":
                raise RefineContractError(
                    f"rule {rule.rule_id} references structure without a typed "
                    "profile-auxiliary selection role"
                )
            for structure in structures:
                auxiliary_id = structure_bindings.get(str(structure))
                if not isinstance(auxiliary_id, str) or not auxiliary_id:
                    raise RefineContractError(
                        f"rule {rule.rule_id} references unbound structure {structure!r}"
                    )
                expected = ("profile_owned_auxiliary_map", f"profile_auxiliary:{auxiliary_id}")
                if expected not in authority:
                    raise RefineContractError(
                        f"rule {rule.rule_id} structure {structure!r} is detached from "
                        "its profile-owned auxiliary authority"
                    )


def _condition_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, list) else (value,)


def _rule_applies(
    applies_when: dict[str, object],
    *,
    pathology_domain_id: str,
    annotation_profile_id: str,
    primitive_id: str,
    annotation: SkillPackage,
    relevant_labels: set[str],
    case_provenance: dict[str, object] | None,
) -> bool:
    """Match only conditions backed by an explicit execution observation."""

    static_values: dict[str, object] = {
        "pathology_domain_id": pathology_domain_id,
        "annotation_profile_id": annotation_profile_id,
        "primitive": primitive_id,
        "primitive_id": primitive_id,
        "dataset": annotation.capabilities.get("dataset_config_name"),
    }
    for key, observed in static_values.items():
        expected = _condition_values(applies_when.get(key))
        if expected and observed not in expected:
            return False

    background = annotation.capabilities.get("background_policy", {})
    background_label = (
        background.get("canonical_label") if isinstance(background, dict) else None
    )
    expected_labels = _condition_values(applies_when.get("canonical_label"))
    if expected_labels and not set(expected_labels).intersection(
        {*relevant_labels, background_label}
    ):
        return False
    requested_labels = _condition_values(applies_when.get("requested_label"))
    if requested_labels and not set(requested_labels).intersection(relevant_labels):
        return False
    native_labels = _condition_values(applies_when.get("native_label"))
    if native_labels and not (0 in native_labels and isinstance(background, dict)):
        return False
    annotation_qualities = _condition_values(
        applies_when.get("annotation_quality")
    )
    if annotation_qualities:
        observed_quality = (case_provenance or {}).get("annotation_quality")
        if observed_quality not in annotation_qualities:
            return False

    recognized_keys = {
        "pathology_domain_id",
        "annotation_profile_id",
        "primitive",
        "primitive_id",
        "dataset",
        "canonical_label",
        "requested_label",
        "native_label",
        "annotation_quality",
        "structure",
    }
    unknown = sorted(set(applies_when) - recognized_keys)
    if unknown:
        raise RefineContractError(
            "execution rule applies_when references unbound observation axes: "
            + ", ".join(unknown)
        )
    return True


def _mask_constraint_applies(
    applies_when: dict[str, object],
    *,
    pathology_domain_id: str,
    annotation_profile_id: str,
    primitive_id: str,
) -> bool:
    """Filter only conditions knowable before multimodal planning."""

    values = {
        "pathology_domain_id": pathology_domain_id,
        "annotation_profile_id": annotation_profile_id,
        "primitive": primitive_id,
        "primitive_id": primitive_id,
    }
    for key, observed in values.items():
        expected = applies_when.get(key)
        if expected is None:
            continue
        allowed = expected if isinstance(expected, list) else [expected]
        if observed not in allowed:
            return False
    return True

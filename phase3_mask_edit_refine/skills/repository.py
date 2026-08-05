"""Discovery, validation, and dual-axis composition for runtime skills."""

from __future__ import annotations

import json
from collections.abc import Iterable
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
        return MaskProfileSchema.from_reference_profile(dataset_config_name)

    def compose(
        self,
        *,
        pathology_domain_id: str,
        annotation_profile_id: str,
        primitive_id: str,
        production: bool,
        available_checker_ids: Iterable[str] = (),
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

        active_rules = tuple(rule for package in packages for rule in package.rules)
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

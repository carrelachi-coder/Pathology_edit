#!/usr/bin/env python3
"""Migrate non-Breast joint skills to certified mask-only Planner authority.

This is an idempotent catalog maintenance command.  It deliberately edits only
the Planner-policy and execution-recognition surfaces; pathology evidence and
render-only limitations remain independent authorities.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

BREAST_DOMAIN = "breast-invasive-carcinoma-v1"
NON_BREAST_MASK_SKILLS = {
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
# This is a positive execution-authority allowlist.  Every other pathology or
# profile rule is documentation for reader-only post-generation QA unless it
# gains a profile-owned typed auxiliary map and a structure-specific checker.
MASK_EXECUTION_RULE_IDS = {
    "glas.require_instance_and_field_grade_provenance",
    "glas.background_is_not_non_gland_tissue",
    "glas.no_implicit_field_grade_change",
    "panda.provider_contract_is_not_interchangeable",
    "panda.background_unknown_is_immutable",
    "panda.burden_edit_preserves_gleason_fine_ids",
    "panda.sparse_masks_do_not_license_topology_repair",
    "ignite.require_exact_native_to_unified_remap",
    "ignite.unannotated_and_background_are_immutable",
    "ignite.condition_on_source_site_and_specimen",
    "puma.condition_on_primary_metastatic_and_site",
    "puma.white_background_is_immutable",
    "puma.immune_is_not_a_tissue_region_label",
    "puma.preserve_tissue_component_topology",
    "orca.fragmented_non_tissue_is_immutable",
    "orca.no_new_tissue_islands_holes_or_bridges",
    "orca.require_profile_identity_and_remap_provenance",
    "prostate.no_implicit_gleason_transition",
    "primitive.generic_immune_infiltrate_decrease.transition",
    "primitive.generic_immune_infiltrate_decrease.interface",
    "primitive.generic_immune_infiltrate_increase.transition",
    "primitive.generic_immune_infiltrate_increase.interface",
    "primitive.necrosis_appearance.transition",
    "primitive.necrosis_appearance.interface",
    "primitive.necrosis_resolution.transition",
    "primitive.necrosis_resolution.interface",
    "primitive.stroma_increase.transition",
    "primitive.stroma_increase.interface",
    "primitive.tumor_decrease.transition",
    "primitive.tumor_decrease.no_deep_notch",
    "primitive.tumor_increase.transition",
    "primitive.tumor_increase.interface",
}
EXECUTION_HE_PATTERN = re.compile(
    r"(?:\bH\s*&\s*E\b|\bH&E\b|source[_ -]?he\b|raw histology)",
    flags=re.IGNORECASE,
)
ALLOWED_OBSERVATIONS = [
    "instruction",
    "semantic_intent",
    "tissue_mask",
    "nuclei_mask",
    "scene_graph",
    "candidate_certificate",
    "skill_rules",
    "user_roi",
    "auxiliary_masks",
]
PROHIBITED_OBSERVATIONS = [
    "source_he_for_execution",
    "unannotated_histology_inference",
]
ALLOWED_DECISIONS = [
    "select_primitive_mechanism_pair",
    "select_certified_tissue_plan_candidate",
    "select_certified_cell_plan_candidate",
    "select_certified_interface_anchor_ids",
    "select_allowed_tool_program",
    "request_clarification",
    "abstain",
]
HARD_CHECKERS = [
    "tissue_gate_binding",
    "whole_instance_changes",
    "joint_provenance",
]
SELECTION_PREFERENCES = [
    "pref:certificate:maximize-capacity-margin",
    "pref:topology:minimize-structural-risk",
]
PROFILE_BY_DOMAIN = {
    "colorectal-adenocarcinoma-v1": "glas-gland-v1",
    "prostate-adenocarcinoma-v1": "panda-gleason-v1",
    "lung-carcinoma-v1": "ignite-semantic-v1",
    "melanoma-v1": "puma-semantic-v1",
    "oral-squamous-cell-carcinoma-v1": "orca-semantic-v1",
}
SIMPLE_INSTRUCTION_BY_PRIMITIVE = {
    "architecture-progression-v1": "Progress the Gleason architectural pattern.",
    "cell-type-abundance-decrease-v1": "Decrease immune cells in the selected region.",
    "cell-type-abundance-increase-v1": "Increase immune cells in the selected region.",
    "cellularity-decrease-v1": "Decrease local cellularity.",
    "cellularity-increase-v1": "Increase local cellularity.",
    "cohesive-boundary-expansion-v1": "Expand the tumor boundary locally.",
    "generic-immune-infiltrate-decrease-v1": "Decrease the generic immune infiltrate.",
    "generic-immune-infiltrate-increase-v1": "Increase the generic immune infiltrate.",
    "generic-inflammatory-cell-abundance-decrease-v1": "Decrease generic inflammatory-cell abundance in the selected region.",
    "generic-inflammatory-cell-abundance-increase-v1": "Increase generic inflammatory-cell abundance in the selected region.",
    "infiltrative-nest-cord-extension-v1": "Add a narrow connected tumor cord.",
    "invasive-front-expansion-v1": "Expand the invasive front.",
    "invasive-tumor-footprint-decrease-v1": "Simulate a post-treatment response by decreasing tumor area.",
    "local-invasive-clearance-v1": "Clear tumor in this local ROI.",
    "necrosis-appearance-v1": "Increase tumor necrosis.",
    "necrosis-resolution-v1": "Reduce tumor necrosis.",
    "neoplastic-cell-abundance-decrease-v1": "Decrease neoplastic cells.",
    "neoplastic-cell-abundance-increase-v1": "Increase neoplastic cells.",
    "neoplastic-microinfiltration-increase-v1": "Increase tumor budding.",
    "peritumoral-neoplastic-scatter-increase-v1": "Add scattered tumor cells near the tumor boundary.",
    "peritumoral-small-cluster-increase-v1": "Add peritumoral small tumor-cell clusters.",
    "residual-tumor-fragmentation-v1": "Simulate post-treatment residual disease by fragmenting residual tumor into controlled foci.",
    "stroma-increase-v1": "Simulate a post-treatment response by increasing operational stroma.",
    "structural-void-spread-v1": "Simulate STAS-like airspace spread.",
    "tumor-burden-increase-v1": "Increase tumor burden.",
}

RECOGNITION_REPLACEMENTS = (
    (r"H&E-confirmed", "mask-component-certified"),
    (r"high-confidence H&E evidence", "digest-bound mask-graph evidence"),
    (r"H&E evidence", "digest-bound mask-graph evidence"),
    (r"H&E-supported", "compiler-certified"),
    (r"visible source architecture", "auxiliary-map-certified source architecture"),
    (r"morphology is visible", "geometry is certified by the selected mask program"),
    (r"true compatible stroma", "digest-bound receiving-validity support"),
    (r"true adjacent stroma", "a legal annotation-profile receiving zone"),
    (r"true stromal receiving compartment", "an explicit Stroma-label receiving compartment"),
    (r"verified airspace", "digest-bound auxiliary airspace"),
    (r"verified septal growth", "digest-bound auxiliary septal support"),
    (r"verified acinar lumen or papillary core", "digest-bound lumen or core auxiliary support"),
    (r"verified colorectal invasive front", "native malignant-gland exterior boundary"),
    (r"verified epidermal/junctional relationship", "explicit Epidermis-label junction relationship"),
    (r"verified invasive-front halo", "compiler-certified pericarcinoma annulus"),
    (r"verified cohesive neoplastic nest or sheet", "explicit Tumor-label component and exterior boundary"),
    (r"verified cohesive squamous nest or cord", "explicit Carcinoma-label component and exterior boundary"),
    (r"verified discrete malignant glands", "native Pattern-3 gland-unit instances"),
    (r"verified pattern-4 fine label", "explicit fine-9 Pattern-4 label"),
    (r"verified pattern-5 fine label or clear non-gland-forming source", "explicit fine-10 Pattern-5 label"),
    (r"source region has verified Gleason pattern 5 authority", "source region has explicit fine-10 Pattern-5 authority"),
    (r"verified", "digest-bound"),
)

SKILL_REPLACEMENTS = (
    (
        "the required H&E, label-profile, native-structure and nuclei observations",
        "the required mask-profile, native-structure, nuclei, and certified-candidate observations",
    ),
    (
        "H&E and available gland structure support a continuous malignant gland and a true compatible stromal interface",
        "native gland-instance, lumen, and receiving-validity authorities support a complete malignant gland unit",
    ),
    ("H&E-supported", "compiler-certified"),
    ("visible unedited boundary sector", "measurable unedited boundary sector"),
    (
        "H&E supports isolated neoplastic cells or clusters of up to four cells at a verified invasive front",
        "a native malignant-gland exterior annulus supports a non-diagnostic one-to-four-cell synthetic representation",
    ),
    ("high-confidence H&E evidence", "digest-bound outer-boundary and annulus certificates"),
    ("visually verified Other-tissue", "mask-component-defined Other-tissue"),
    ("sufficiently long visible,", "sufficiently long mask-defined,"),
    ("bounded H&E rendering", "bounded post-generation rendering"),
    ("verified", "annotation-authorized"),
)


def _rewrite_strings(value: Any) -> Any:
    if isinstance(value, str):
        for pattern, replacement in RECOGNITION_REPLACEMENTS:
            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [_rewrite_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_strings(item) for key, item in value.items()}
    return value


def _planner_policy() -> dict[str, Any]:
    return {
        "allowed_observation_sources": list(ALLOWED_OBSERVATIONS),
        "prohibited_observation_sources": list(PROHIBITED_OBSERVATIONS),
        "hard_constraint_checker_ids": list(HARD_CHECKERS),
        "selection_preferences": list(SELECTION_PREFERENCES),
        "clarification_triggers": [
            "different surviving primitive hypotheses encode different edit scopes",
            "a requested diagnostic entity is representable only as a non-diagnostic synthetic downgrade",
            "a treatment-specific mechanism lacks explicit treatment direction",
        ],
        "allowed_decisions": list(ALLOWED_DECISIONS),
    }


def _reader_only_pathology_fact(rule: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(rule)
    migrated["scope"] = "reader_only_pathology_fact"
    migrated["severity"] = "advisory"
    migrated["deterministic_check_id"] = None
    migrated["critic_requirement"] = None
    migrated["execution_role"] = None
    migrated["observation_authority"] = []
    migrated.pop("selection_preference", None)
    migrated["preference_rule_id"] = None
    migrated["required_observation"] = (
        "Counterfactual histology may be inspected only after generation in "
        "reader-only QA; this fact is unavailable to execution selection or veto."
    )
    limitations = list(migrated.get("known_limitations", ()))
    statement = (
        "Reader-only pathology facts are excluded from the execution knowledge bundle."
    )
    if statement not in limitations:
        limitations.append(statement)
    migrated["known_limitations"] = limitations
    return migrated


def _typed_execution_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Replace prose observation authority with a closed typed contract."""

    migrated = dict(rule)
    checker_id = str(migrated.get("deterministic_check_id") or "").strip()
    if not checker_id:
        raise ValueError(
            f"execution rule {migrated.get('rule_id')} has no deterministic checker"
        )
    if checker_id == "profile_required_provenance":
        role = "provenance_precondition"
        authority = [
            {
                "source": "case_provenance",
                "binding": "profile_required_provenance",
            },
            {
                "source": "deterministic_metric",
                "binding": f"checker:{checker_id}",
            },
        ]
    elif checker_id == "semantic_capability_guard":
        role = "semantic_capability_precondition"
        authority = [
            {
                "source": "instruction_semantic_intent",
                "binding": "primitive_id",
            },
            {
                "source": "tissue_mask",
                "binding": "source_mask_sha256",
            },
            {
                "source": "deterministic_metric",
                "binding": f"checker:{checker_id}",
            },
        ]
    else:
        role = "deterministic_mask_invariant"
        authority = [
            {
                "source": "tissue_mask",
                "binding": "source_mask_sha256",
            },
            {
                "source": "scene_graph",
                "binding": "compiler_scene_graph",
            },
            {
                "source": "deterministic_metric",
                "binding": f"checker:{checker_id}",
            },
        ]
    migrated["execution_role"] = role
    migrated["observation_authority"] = authority
    migrated.pop("selection_preference", None)
    migrated["preference_rule_id"] = None
    if migrated.get("rule_id") == "panda.sparse_masks_do_not_license_topology_repair":
        # Sparse PANDA annotation never authorizes the editor to invent or repair
        # topology.  This is an unconditional profile constraint; activation may
        # not depend on a caller-provided, untyped annotation_quality value.
        migrated["applies_when"] = {
            "annotation_profile_id": "panda-gleason-v1"
        }
    migrated["critic_requirement"] = None
    migrated["claim"] = (
        f"Enforce the compiler-bound {checker_id} execution invariant."
    )
    migrated["required_observation"] = (
        "Use only the typed observation_authority records; no unannotated "
        "histologic structure observation is authorized."
    )
    migrated["exceptions"] = []
    migrated["counterexamples"] = []
    migrated["expected_morphology"] = []
    migrated["forbidden_morphology"] = []
    return migrated


def _sanitize_mask_contract(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    constraints = []
    for raw in payload.get("constraints", ()):
        constraint = _rewrite_strings(dict(raw))
        constraint["observability"] = [
            item
            for item in constraint.get("observability", ())
            if item != "source_he"
        ]
        if not constraint["observability"]:
            constraint["observability"] = ["auxiliary_structure_map"]
        constraint["critic_requirement"] = None
        constraint["required_inputs"] = [
            (
                "digest-bound native/auxiliary structure map"
                if EXECUTION_HE_PATTERN.search(str(item))
                else item
            )
            for item in constraint.get("required_inputs", ())
        ]
        statement = str(constraint.get("mask_statement", ""))
        statement = re.sub(
            r"if visible only on H&E, mask edit cannot guarantee preservation and must abstain on ambiguous overlap",
            "without a digest-bound native/auxiliary map, that structure is unavailable to execution and the affected candidate must abstain",
            statement,
            flags=re.IGNORECASE,
        )
        statement = re.sub(
            r"H&E recognition alone supports Planner veto, not a mask guarantee",
            "unencoded structure recognition is unavailable to the execution Planner",
            statement,
            flags=re.IGNORECASE,
        )
        statement = EXECUTION_HE_PATTERN.sub(
            "digest-bound native/auxiliary structure authority", statement
        )
        constraint["mask_statement"] = statement
        constraints.append(constraint)
    migrated["constraints"] = constraints
    return migrated


def _sanitize_mask_rules(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    rules = []
    for raw in payload.get("rules", ()):
        rule = dict(raw)
        if str(rule.get("scope", "")).startswith("reader_only_"):
            rule = _reader_only_pathology_fact(rule)
        elif rule.get("rule_id") in MASK_EXECUTION_RULE_IDS:
            rule = _typed_execution_rule(rule)
        else:
            rule = _reader_only_pathology_fact(rule)
        rules.append(rule)
    migrated["rules"] = rules
    return migrated


def _sanitize_mask_skill_markdown(text: str) -> str:
    lines = text.splitlines()
    try:
        second_boundary = lines.index("---", 1)
    except ValueError:
        second_boundary = 3
    frontmatter = "\n".join(lines[: second_boundary + 1])
    name = next(
        (
            line.split(":", 1)[1].strip()
            for line in lines[: second_boundary + 1]
            if line.startswith("name:")
        ),
        "non-breast-mask-skill",
    )
    return (
        frontmatter
        + "\n\n# "
        + name
        + " mask-only execution authority\n\n"
        + "1. Read `references/mask_contract.json` first and enforce only "
        + "digest-bound provenance, semantic masks, scene graphs, native annotations, "
        + "auxiliary structure maps, and deterministic candidate certificates.\n"
        + "2. Raw histology, overlays, crops, reader boards, and renamed image panels "
        + "are unavailable to the execution Planner and Critic.\n"
        + "3. Unencoded glands, lumina, epidermis, lung structures, invasive fronts, "
        + "grades, and treatment effects cannot be inferred for anchor selection or veto.\n"
        + "4. Rules marked `reader_only_pathology_fact` may guide post-generation "
        + "reader QA only and are excluded from the execution knowledge bundle.\n"
        + "5. Abstain whenever the required native or auxiliary authority is absent; "
        + "all capabilities remain draft and shadow-only.\n"
    )


def _migrate_mask_skill_authority(root: Path, *, check: bool) -> list[Path]:
    changed: list[Path] = []
    catalog = root / "phase3_mask_edit_refine" / "skills" / "catalog"
    for kind in ("annotation-profile", "pathology-domain"):
        for skill_id in sorted(NON_BREAST_MASK_SKILLS):
            base = catalog / kind / skill_id
            if not base.is_dir():
                continue
            for filename, sanitizer in (
                ("rules.json", _sanitize_mask_rules),
                ("mask_contract.json", _sanitize_mask_contract),
            ):
                path = base / "references" / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                rendered = json.dumps(
                    sanitizer(payload), indent=2, ensure_ascii=False
                ) + "\n"
                if rendered != path.read_text(encoding="utf-8"):
                    changed.append(path)
                    if not check:
                        path.write_text(rendered, encoding="utf-8")
            skill_path = base / "SKILL.md"
            rendered_skill = _sanitize_mask_skill_markdown(
                skill_path.read_text(encoding="utf-8")
            )
            if rendered_skill != skill_path.read_text(encoding="utf-8"):
                changed.append(skill_path)
                if not check:
                    skill_path.write_text(rendered_skill, encoding="utf-8")
    for base in sorted((catalog / "edit-primitive").iterdir()):
        rules_path = base / "references" / "rules.json"
        if not rules_path.is_file():
            continue
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
        rendered = json.dumps(
            _sanitize_mask_rules(payload), indent=2, ensure_ascii=False
        ) + "\n"
        if rendered != rules_path.read_text(encoding="utf-8"):
            changed.append(rules_path)
            if not check:
                rules_path.write_text(rendered, encoding="utf-8")
    return changed


def migrate(root: Path, *, check: bool) -> list[Path]:
    changed: list[Path] = _migrate_mask_skill_authority(root, check=check)
    mechanism_root = root / "phase3_joint_edit_refine" / "skills" / "catalog" / "joint-mechanism"
    for path in sorted(mechanism_root.glob("*/references/joint_contract.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("pathology_domain_id") == BREAST_DOMAIN:
            continue
        migrated = dict(payload)
        if not isinstance(migrated.get("planner_policy"), dict):
            migrated["planner_policy"] = _planner_policy()
        recognition = migrated.get("recognition_contract")
        if isinstance(recognition, dict):
            migrated["recognition_contract"] = _rewrite_strings(recognition)
        rendered = json.dumps(migrated, indent=2, ensure_ascii=False) + "\n"
        if rendered != path.read_text(encoding="utf-8"):
            changed.append(path)
            if not check:
                path.write_text(rendered, encoding="utf-8")

        skill_path = path.parents[1] / "SKILL.md"
        skill_text = skill_path.read_text(encoding="utf-8")
        migrated_skill = skill_text
        for old, new in SKILL_REPLACEMENTS:
            migrated_skill = migrated_skill.replace(old, new)
        if migrated_skill != skill_text:
            changed.append(skill_path)
            if not check:
                skill_path.write_text(migrated_skill, encoding="utf-8")

        evidence_path = path.with_name("evidence.json")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        migrated_evidence = dict(evidence)
        migrated_records = []
        for record in evidence.get("records", []):
            migrated_record = dict(record)
            if record.get("authority_category") == "engineering_proxy":
                migrated_record["claim_scope"] = sorted(
                    {*record.get("claim_scope", []), "planner_policy"}
                )
            migrated_records.append(migrated_record)
        migrated_evidence["records"] = migrated_records
        rendered_evidence = (
            json.dumps(migrated_evidence, indent=2, ensure_ascii=False) + "\n"
        )
        if rendered_evidence != evidence_path.read_text(encoding="utf-8"):
            changed.append(evidence_path)
            if not check:
                evidence_path.write_text(rendered_evidence, encoding="utf-8")
    matrix_path = (
        root
        / "phase3_joint_edit_refine"
        / "resources"
        / "non_breast_organ_annotation_capability_matrix_v1.json"
    )
    matrix_text = _render_capability_matrix(root)
    existing_matrix = (
        matrix_path.read_text(encoding="utf-8") if matrix_path.exists() else ""
    )
    if matrix_text != existing_matrix:
        changed.append(matrix_path)
        if not check:
            matrix_path.write_text(matrix_text, encoding="utf-8")
    return changed


def _render_capability_matrix(root: Path) -> str:
    import sys

    sys.path.insert(0, str(root))
    from phase3_joint_edit_refine.skills.repository import JointSkillRepository

    repository = JointSkillRepository(root / "phase3_joint_edit_refine" / "skills" / "catalog")
    profiles = []
    for domain_id, profile_id in PROFILE_BY_DOMAIN.items():
        annotation = repository.annotation_profiles[profile_id]
        capabilities = []
        for mechanism in sorted(
            (
                item
                for item in repository.mechanisms.values()
                if item.pathology_domain_id == domain_id
            ),
            key=lambda item: item.mechanism_id,
        ):
            for primitive_id in mechanism.supported_primitives:
                closed_reason = repository.execution_selection_reason(
                    primitive_id=primitive_id,
                    mechanism_id=mechanism.mechanism_id,
                )
                if closed_reason is not None:
                    status = "closed"
                elif (
                    mechanism.mechanism_id in annotation.conditional_mechanisms
                    or mechanism.representability.status == "conditionally_supported"
                    or mechanism.representability.required_auxiliary_structures
                ):
                    status = "conditionally_supported"
                else:
                    status = "shadow_only"
                capabilities.append(
                    {
                        "primitive_id": primitive_id,
                        "mechanism_id": mechanism.mechanism_id,
                        "status": status,
                        "production_status": "shadow_only",
                        "simple_instructions": [
                            SIMPLE_INSTRUCTION_BY_PRIMITIVE[primitive_id]
                        ],
                        "required_auxiliary_structures": list(
                            mechanism.representability.required_auxiliary_structures
                        ),
                        "closed_reason": closed_reason,
                    }
                )
        profiles.append(
            {
                "pathology_domain_id": domain_id,
                "annotation_profile_id": profile_id,
                "production_status": "shadow_only",
                "planner_selection_contract": {
                    "tissue": {
                        "decision_id": "select_certified_tissue_plan_candidate",
                        "selectable_fields": [
                            "selected_candidate_id",
                            "selected_tool_family",
                            "supporting_preference_rule_ids",
                        ],
                    },
                    "cell": {
                        "decision_id": "select_certified_cell_plan_candidate",
                        "selectable_fields": [
                            "selected_candidate_id",
                            "selected_tool_program_id",
                            "supporting_preference_rule_ids",
                        ],
                    },
                    "compiler_owned_fields": [
                        "interfaces",
                        "components",
                        "anchors",
                        "zones",
                        "annuli",
                        "layouts",
                        "parameter_values",
                        "parameter_ranges",
                        "area",
                        "depth",
                        "geometry",
                        "cell_counts",
                        "coordinates",
                        "pixels",
                    ],
                },
                "capabilities": capabilities,
            }
        )
    return (
        json.dumps(
            {
                "schema_version": "non-breast-organ-annotation-capability-matrix-v1",
                "production_status": "shadow_only",
                "profiles": profiles,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    changed = migrate(args.root, check=args.check)
    if args.check and changed:
        for path in changed:
            print(path.relative_to(args.root))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

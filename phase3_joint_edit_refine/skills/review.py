"""Generate a human-reviewable summary without promoting draft knowledge."""

from __future__ import annotations

import json
from pathlib import Path

from .repository import JointSkillRepository


def build_joint_knowledge_review(repository: JointSkillRepository | None = None) -> str:
    repository = repository or JointSkillRepository()
    lines = [
        "# Joint Pathology Knowledge Review",
        "",
        "> Evidence is governed by four non-interchangeable authorities. Draft review, uncalibrated statistics or pending generator representability keep production fail-closed.",
        "",
        "## Evidence authority boundary",
        "",
        "| Authority | Owns | Must not authorize |",
        "|---|---|---|",
        "| pathology_fact | Disease morphology, mechanism recognition, contraindications and pathology counterexamples | Dataset label semantics, numeric tool thresholds or generator capability |",
        "| dataset_fact | Label ontology, unannotated/background meaning, fine IDs, revision and digest | Biological mechanism or cross-dataset behavior |",
        "| engineering_proxy | Tool programs, capacity estimates, geometry thresholds and deterministic gates | A pathology identity or successful H&E rendering |",
        "| model_representability | What the frozen nuclei/H&E condition stack has demonstrated in paired evaluation | Pathology truth or annotation semantics |",
        "",
        "The runtime rejects any unclassified top-level contract field and any source used under the wrong authority category.",
        "",
        "## Primitive inventory",
        "",
        "| Primitive | Scope | Tissue action | Budget | Baselines | Quota role | Mechanism coverage |",
        "|---|---|---|---|---|---|---:|",
    ]
    for item in sorted(
        repository.primitives.values(), key=lambda value: value.primitive_id
    ):
        coverage = sum(
            item.primitive_id in mechanism.supported_primitives
            for mechanism in repository.mechanisms.values()
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    item.primitive_id,
                    item.scope,
                    item.tissue_action,
                    item.budget_mode,
                    ", ".join(item.allowed_baseline_modes),
                    ", ".join(item.allowed_quota_roles),
                    str(coverage),
                )
            )
            + " |"
        )
    lines.extend([
        "",
        "A zero mechanism coverage is an explicit fail-closed research placeholder, not an executable claim.",
        "",
        "## Mechanism inventory",
        "",
        "| Domain | Mechanism | Tissue program | Cell layouts | Auxiliary observations | Render-only claims |",
        "|---|---|---|---|---|---|",
    ])
    for item in sorted(repository.mechanisms.values(), key=lambda value: (value.pathology_domain_id, value.mechanism_id)):
        lines.append(
            "| " + " | ".join(
                (
                    item.pathology_domain_id,
                    item.mechanism_id,
                    item.tissue_program.mode,
                    ", ".join(item.cell_program.layout_programs),
                    ", ".join(item.representability.required_auxiliary_structures) or "none",
                    ", ".join(item.render.render_only_claims) or "none",
                )
            ) + " |"
        )
    lines.extend([
        "",
        "## Annotation profile protections",
        "",
        "| Profile | Cell-prohibited IDs | G-prohibited IDs | Required provenance | Conditional mechanisms |",
        "|---|---|---|---|---|",
    ])
    for item in sorted(repository.annotation_profiles.values(), key=lambda value: value.annotation_profile_id):
        lines.append(
            "| " + " | ".join(
                (
                    item.annotation_profile_id,
                    ", ".join(map(str, item.prohibit_cell_placement_fine_ids)),
                    ", ".join(map(str, item.prohibit_generation_support_fine_ids)),
                    ", ".join(item.required_provenance_fields),
                    ", ".join(item.conditional_mechanisms) or "none",
                )
            ) + " |"
        )
    matrix = repository.capability_matrix()
    counts = {}
    for row in matrix:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines.extend([
        "",
        "## Capability matrix summary",
        "",
        f"The matrix contains {len(matrix)} domain×profile×mechanism rows: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        + ". A conditional row still requires case-level H&E observations, auxiliary structures, source-matched cells and all registered gates.",
        "",
        "## Executable mechanism details",
        "",
    ])
    for item in sorted(
        repository.mechanisms.values(),
        key=lambda value: (value.pathology_domain_id, value.mechanism_id),
    ):
        evidence_status = repository.skill_evidence_status[
            f"joint-mechanism:{item.mechanism_id}"
        ]
        transitions = []
        for primitive_id, contract in sorted(
            item.tissue_program.primitive_label_contracts.items()
        ):
            transitions.append(
                f"`{primitive_id}`: {', '.join(contract['source_labels'])} → "
                f"{', '.join(contract['target_labels'])}"
            )
        lines.extend(
            [
                f"### {item.mechanism_id}",
                "",
                f"- Domain/status/version: `{item.pathology_domain_id}` / `{item.review_status}` / `{item.version}`",
                f"- Summary: {item.summary}",
                "- Required H&E/scene observations: "
                + "; ".join(item.recognition.required_observations),
                "- Contraindications: "
                + ("; ".join(item.recognition.contraindications) or "none"),
                f"- Representability: `{item.representability.status}`; required cell classes "
                + ", ".join(map(str, item.representability.required_cell_classes))
                + "; auxiliary maps "
                + (", ".join(item.representability.required_auxiliary_structures) or "none")
                + f"; semantic-instance fallback={item.representability.allow_semantic_instance_fallback}",
                f"- Tissue program: `{item.tissue_program.mode}`; tools "
                + ", ".join(item.tissue_program.allowed_tools),
                "- Primitive label contracts: " + "; ".join(transitions),
                "- Protected/prohibited structures: "
                + ("; ".join(item.tissue_program.prohibited_structures) or "none"),
                "- Cell execution: actions "
                + ", ".join(item.cell_program.actions)
                + "; layouts "
                + ", ".join(item.cell_program.layout_programs)
                + f"; core `{item.cell_program.core_policy}`; halo `{item.cell_program.halo_policy}` "
                + f"{item.cell_program.halo_distance_px}px; cluster {item.cell_program.cluster_size_range}",
                f"- Coupling/area: `{item.coupling.joint_area_mode}`; tissue floor="
                + str(item.coupling.tissue_floor_applies)
                + f"; planned cell-only share={item.coupling.cell_only_target_fraction:.3f}; "
                + f"render support `{item.coupling.render_support_policy_id}`",
                "- Required deterministic gates: "
                + ", ".join(
                    dict.fromkeys(
                        [
                            *item.tissue_program.required_checker_ids,
                            *item.cell_program.required_checker_ids,
                            *item.joint_gate_ids,
                        ]
                    )
                ),
                "- Mask-level guarantees: "
                + ("; ".join(item.render.mask_guarantees) or "none"),
                "- Render/critic vetoes: " + "; ".join(item.render.veto_findings),
                "- Render-only claims: "
                + ("; ".join(item.render.render_only_claims) or "none"),
                "- Counterexamples: " + "; ".join(item.counterexamples),
                "- Governed evidence sources: "
                + "; ".join(evidence_status.source_ids),
                "- Evidence category status: "
                + "; ".join(
                    f"{key}={value}"
                    for key, value in sorted(
                        evidence_status.category_status.items()
                    )
                ),
                "",
            ]
        )
    lines.extend([
        "## Profile-level fail-closed contracts",
        "",
    ])
    for item in sorted(
        repository.annotation_profiles.values(),
        key=lambda value: value.annotation_profile_id,
    ):
        fine = "; ".join(
            f"{mechanism}: {list(ids)}"
            for mechanism, ids in sorted(item.mechanism_required_fine_ids.items())
        ) or "none"
        lines.extend(
            [
                f"### {item.annotation_profile_id}",
                "",
                f"- Status/version: `{item.review_status}` / `{item.version}`",
                f"- Tissue/joint prohibited IDs: {list(item.prohibited_fine_ids)}",
                f"- Cell-placement prohibited IDs: {list(item.prohibit_cell_placement_fine_ids)}",
                f"- Generation-support prohibited IDs: {list(item.prohibit_generation_support_fine_ids)}",
                "- Required provenance: " + ", ".join(item.required_provenance_fields),
                "- Required gates: " + ", ".join(item.required_checker_ids),
                "- Unavailable mechanisms: "
                + (", ".join(item.unavailable_mechanisms) or "none"),
                "- Conditional mechanisms: "
                + (", ".join(item.conditional_mechanisms) or "none"),
                "- Mechanism-specific fine-label binding: " + fine,
                "",
            ]
        )
    lines.extend([
        "## Evidence and review boundary",
        "",
        "The catalog-level `evidence-governance-v2.json` classifies every contract field and binds pathology domains and annotation profiles to separate sources. Each atomic mechanism also keeps `joint_contract.json`, `evidence.json`, `counterexamples.json` and `statistics.json`. Render-only claims cannot be promoted to mask guarantees. Production requires verified applicable sources, calibrated cohort statistics, frozen generator-response evidence and internal pathology/engineering review.",
        "",
        "## Evidence gaps by skill",
        "",
    ])
    for key, status in sorted(repository.skill_evidence_status.items()):
        lines.append(
            f"- `{key}`: production_allowed={status.production_allowed}; "
            + ("; ".join(status.gaps) if status.gaps else "no evidence gap")
        )
    return "\n".join(lines) + "\n"


def write_joint_knowledge_review(
    path: str | Path,
    repository: JointSkillRepository | None = None,
) -> None:
    Path(path).write_text(
        build_joint_knowledge_review(repository), encoding="utf-8"
    )


def export_capability_matrix(path: str | Path, repository: JointSkillRepository | None = None) -> None:
    repository = repository or JointSkillRepository()
    Path(path).write_text(json.dumps(repository.capability_matrix(), indent=2, sort_keys=True), encoding="utf-8")

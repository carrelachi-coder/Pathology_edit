#!/usr/bin/env python3
"""Install the shadow-only non-Breast organ/annotation primitive lattice.

The catalog is deliberately data driven.  This maintenance command keeps the
large set of profile/mechanism contracts deterministic and provides ``--check``
for drift tests.  It never grants production or H&E execution authority.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

CATALOG = Path("phase3_joint_edit_refine/skills/catalog")
MECHANISMS = CATALOG / "joint-mechanism"
PROFILES = CATALOG / "annotation-profile"
PRIMITIVES = CATALOG / "edit-primitive"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _ordered_union(values: list[Any], additions: list[Any]) -> list[Any]:
    return [*values, *(item for item in additions if item not in values)]


class Writer:
    def __init__(self, root: Path, *, check: bool) -> None:
        self.root = root
        self.check = check
        self.changed: list[Path] = []

    def text(self, relative: Path, value: str) -> None:
        path = self.root / relative
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current == value:
            return
        self.changed.append(path)
        if not self.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")

    def json(self, relative: Path, value: Any) -> None:
        self.text(relative, _render(value))


def _contract(root: Path, mechanism_id: str) -> dict[str, Any]:
    return _load(
        root / MECHANISMS / mechanism_id / "references" / "joint_contract.json"
    )


def _write_contract(writer: Writer, payload: dict[str, Any]) -> None:
    mechanism_id = payload["mechanism_id"]
    writer.json(
        MECHANISMS / mechanism_id / "references" / "joint_contract.json",
        payload,
    )


def _write_shadow_skill(
    writer: Writer,
    mechanism_id: str,
    *,
    title: str,
    body: str,
    display_name: str | None = None,
) -> None:
    base = MECHANISMS / mechanism_id
    writer.text(
        base / "SKILL.md",
        "---\n"
        f"name: {mechanism_id}\n"
        f"description: {title}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body.strip()}\n",
    )
    writer.text(
        base / "agents" / "openai.yaml",
        "interface:\n"
        f'  display_name: "{display_name or mechanism_id}"\n'
        '  short_description: "Shadow-only certified mask mechanism"\n'
        f'  default_prompt: "Use ${mechanism_id} only to select compiler-certified mask candidates; never infer execution authority from H&E."\n',
    )


def _add_primitives(
    contract: dict[str, Any],
    *,
    label_contracts: dict[str, dict[str, list[str]]],
    layouts: dict[str, str],
) -> None:
    contract["supported_primitives"] = _ordered_union(
        contract["supported_primitives"], list(label_contracts)
    )
    tissue = contract["tissue_program"]
    tissue["primitive_label_contracts"].update(label_contracts)
    cell = contract["cell_program"]
    cell["layout_program_by_primitive"].update(layouts)
    cell["layout_programs"] = _ordered_union(
        cell["layout_programs"], list(dict.fromkeys(layouts.values()))
    )


def _add_policy_checks(contract: dict[str, Any], checks: list[str]) -> None:
    contract["planner_policy"]["hard_constraint_checker_ids"] = _ordered_union(
        contract["planner_policy"]["hard_constraint_checker_ids"], checks
    )
    contract["tissue_program"]["required_checker_ids"] = _ordered_union(
        contract["tissue_program"]["required_checker_ids"], checks
    )
    contract["joint_gate_ids"] = _ordered_union(contract["joint_gate_ids"], checks)


def _cell_only_dispersion(
    contract: dict[str, Any],
    *,
    host_label: str,
    include_cluster: bool,
) -> None:
    label_contracts = {
        "peritumoral-neoplastic-scatter-increase-v1": {
            "source_labels": [host_label],
            "target_labels": [host_label],
        }
    }
    layouts = {"peritumoral-neoplastic-scatter-increase-v1": "single"}
    if include_cluster:
        label_contracts["peritumoral-small-cluster-increase-v1"] = {
            "source_labels": [host_label],
            "target_labels": [host_label],
        }
        layouts["peritumoral-small-cluster-increase-v1"] = "small_cluster"
    _add_primitives(contract, label_contracts=label_contracts, layouts=layouts)
    contract["coupling_contract"]["allow_neoplastic_in_non_tumor_tissue"] = True
    _add_policy_checks(
        contract,
        [
            "external_boundary_binding",
            "peritumoral_annulus",
        ],
    )
    for field in (
        contract["planner_policy"]["hard_constraint_checker_ids"],
        contract["tissue_program"]["required_checker_ids"],
        contract["joint_gate_ids"],
    ):
        field[:] = [
            item for item in field if item not in {"no_remote_focus", "no_bridge_to_primary"}
        ]
    contract["cell_program"]["required_checker_ids"] = _ordered_union(
        contract["cell_program"]["required_checker_ids"],
        ["mechanism_realization"],
    )


def _remove_mixed_scope_dispersion_checks(contract: dict[str, Any]) -> None:
    """Leave primitive-specific dispersion checks inside the postcondition.

    A local-population mechanism also owns abundance/cellularity primitives;
    registering annulus checks as unconditional mechanism gates would reject
    those legal component-zone programs.
    """

    dispersion_checks = {"external_boundary_binding", "peritumoral_annulus"}
    for field in (
        contract["planner_policy"]["hard_constraint_checker_ids"],
        contract["tissue_program"]["required_checker_ids"],
        contract["joint_gate_ids"],
    ):
        field[:] = [item for item in field if item not in dispersion_checks]


def _sanitize_orca_language(value: Any) -> Any:
    """Remove tissue classes and claims that ORCA does not encode."""

    if isinstance(value, str):
        return (
            value.replace("Stroma", "Other tissue")
            .replace("stromal", "connective-tissue")
            .replace("stroma", "a specific connective-tissue compartment")
        )
    if isinstance(value, list):
        result = []
        for item in value:
            sanitized = _sanitize_orca_language(item)
            if sanitized not in result:
                result.append(sanitized)
        return result
    if isinstance(value, dict):
        return {
            key: _sanitize_orca_language(item) for key, item in value.items()
        }
    return value


def _oral_scatter_transform(contract: dict[str, Any]) -> None:
    """Define non-diagnostic ORCA scatter from annotation-only authority."""

    contract["summary"] = (
        "Add annotation-anchored synthetic class-1 singles or 1-4-cell foci "
        "outside an explicit ORCA Tumor component without diagnosing invasion."
    )
    contract["recognition_contract"] = {
        "required_observations": [
            "explicit ORCA Tumor/Other-tissue external mask boundary",
            "compiler-certified bounded outer annulus",
            "source-matched complete class-1 reference nuclei",
        ],
        "contraindications": [
            "fragmented zero/non-tissue or another protected footprint",
            "remote focus or bridge to bulk Tumor",
            "WPOI, budding, invasion or prognostic diagnosis requested",
        ],
        "minimum_confidence": 0.9,
    }
    contract["tissue_program"]["mode"] = (
        "preserve_orca_tissue_for_annotation_anchored_scatter"
    )
    contract["tissue_program"]["prohibited_structures"] = [
        "fragmented_non_tissue",
        "surface_space",
    ]
    contract["cell_program"]["actions"] = ["retain", "add"]
    contract["cell_program"]["allowed_cell_classes"] = [1]
    contract["cell_program"]["layout_programs"] = ["single", "small_cluster"]
    contract["cell_program"]["halo_policy"] = (
        "add_complete_class1_in_certified_tumor_outer_annulus"
    )
    contract["cell_program"]["halo_distance_px"] = [4, 32]
    contract["cell_program"]["cluster_size_range"] = [1, 4]
    contract["coupling_contract"]["compatibility_rule_ids"] = [
        "orca.mask.annotation_anchored_outer_annulus",
        "orca.mask.complete_class1_synthetic_foci",
        "orca.mask.fragmented_zero_preserved",
    ]
    contract["render_contract"] = {
        "required_findings": [
            "bounded synthetic singles or 1-4-cell foci in Other tissue",
            "ORCA Tumor and fragmented zero/non-tissue remain pixel-exact",
        ],
        "veto_findings": [
            "bulk bridge, remote focus or protected-footprint overlap",
            "diagnostic WPOI, budding, invasion or prognostic claim",
        ],
        "mask_guarantees": [
            "tissue mask is pixel-exact",
            "complete class-1 footprints remain inside the certified annulus",
        ],
        "render_only_claims": [
            "WPOI",
            "tumor budding",
            "histologic invasion",
            "prognosis",
        ],
    }
    contract["counterexamples"] = [
        "calling synthetic scatter diagnostic WPOI or budding",
        "placing a focus in fragmented zero/non-tissue",
        "accepting a remote focus or solid bridge",
    ]


def _add_local_cell_primitives(
    contract: dict[str, Any],
    *,
    include_generic_inflammatory: bool,
    generic_host_labels: tuple[str, ...] = (
        "Tumor",
        "Stroma",
        "Other tissue",
    ),
) -> None:
    label_contracts = {
        "neoplastic-cell-abundance-increase-v1": {
            "source_labels": ["Tumor"],
            "target_labels": ["Tumor"],
        },
        "neoplastic-cell-abundance-decrease-v1": {
            "source_labels": ["Tumor"],
            "target_labels": ["Tumor"],
        },
    }
    layouts = {
        "neoplastic-cell-abundance-increase-v1": "small_cluster",
        "neoplastic-cell-abundance-decrease-v1": "localized_density_gradient",
    }
    if include_generic_inflammatory:
        for direction, layout in (
            ("increase", "small_cluster"),
            ("decrease", "localized_density_gradient"),
        ):
            primitive = f"generic-inflammatory-cell-abundance-{direction}-v1"
            label_contracts[primitive] = {
                "source_labels": list(generic_host_labels),
                "target_labels": list(generic_host_labels),
            }
            layouts[primitive] = layout
    _add_primitives(contract, label_contracts=label_contracts, layouts=layouts)


def _add_retreat_primitives(contract: dict[str, Any]) -> None:
    _add_primitives(
        contract,
        label_contracts={
            "invasive-tumor-footprint-decrease-v1": {
                "source_labels": ["Tumor"],
                "target_labels": ["Stroma"],
            },
            "stroma-increase-v1": {
                "source_labels": ["Tumor"],
                "target_labels": ["Stroma"],
            },
            "residual-tumor-fragmentation-v1": {
                "source_labels": ["Tumor"],
                "target_labels": ["Stroma"],
            },
        },
        layouts={
            "invasive-tumor-footprint-decrease-v1": "population_replacement",
            "stroma-increase-v1": "population_replacement",
            "residual-tumor-fragmentation-v1": "population_replacement",
        },
    )
    _add_policy_checks(contract, ["residual_fragmentation_topology"])


def _generic_boundary_growth(contract: dict[str, Any], *, host: str) -> None:
    _add_primitives(
        contract,
        label_contracts={
            "cohesive-boundary-expansion-v1": {
                "source_labels": [host],
                "target_labels": ["Tumor"],
            }
        },
        layouts={"cohesive-boundary-expansion-v1": "boundary_aligned"},
    )
    contract["summary"] = (
        "Expand an existing annotation-defined Tumor component from a certified "
        "external mask boundary; subtype identity is not an execution claim."
    )
    contract["recognition_contract"]["required_observations"] = [
        "explicit Tumor-label component",
        "compiler-certified external receiving interface",
        "source-matched complete neoplastic reference nuclei",
    ]
    contract["recognition_contract"]["contraindications"] = [
        "protected profile label in the edit band",
        "remote island or component bridge",
        "subtype or diagnostic-front claim inferred from H&E",
    ]
    contract["cell_program"]["layout_programs"] = _ordered_union(
        contract["cell_program"]["layout_programs"], ["boundary_aligned"]
    )
    for field in (
        contract["planner_policy"]["hard_constraint_checker_ids"],
        contract["tissue_program"]["required_checker_ids"],
        contract["joint_gate_ids"],
    ):
        field[:] = [
            item
            for item in field
            if item != "annotation_anchored_extension_geometry"
        ]
    _add_policy_checks(contract, ["external_boundary_binding"])


def _remove_redundant_tumor_burden_growth(contract: dict[str, Any]) -> None:
    """Keep one unambiguous external-boundary growth primitive per organ."""

    primitive_id = "tumor-burden-increase-v1"
    contract["supported_primitives"] = [
        item
        for item in contract["supported_primitives"]
        if item != primitive_id
    ]
    contract["tissue_program"]["primitive_label_contracts"].pop(
        primitive_id, None
    )
    contract["cell_program"]["layout_program_by_primitive"].pop(
        primitive_id, None
    )


def _cord(contract: dict[str, Any], *, host: str) -> None:
    contract["supported_primitives"] = ["infiltrative-nest-cord-extension-v1"]
    contract["summary"] = (
        "Create one annotation-anchored narrow connected Tumor extension with "
        "class-1 seam continuity; this is synthetic cord geometry only."
    )
    contract["recognition_contract"] = {
        "required_observations": [
            "explicit Tumor-label component and exterior boundary",
            f"explicit {host} receiving label",
            "compiler-certified tapered projection capacity",
        ],
        "contraindications": [
            "protected structure in projection corridor",
            "remote island or side merge",
            "histologic invasive-front diagnosis requested",
        ],
        "minimum_confidence": 0.9,
    }
    tissue_program = contract["tissue_program"]
    tissue_program["mode"] = "annotation_anchored_narrow_connected_extension"
    tissue_program["target_component_merge_policy"] = "selected_only"
    tissue_program["primitive_label_contracts"] = {
        "infiltrative-nest-cord-extension-v1": {
            "source_labels": [host],
            "target_labels": ["Tumor"],
        }
    }
    tissue_program["allowed_tools"] = ["directional_tapered_projection"]
    tissue_program["required_checker_ids"] = [
        "tissue_gate_binding",
        "external_boundary_binding",
        "annotation_anchored_extension_geometry",
        "profile_fine_transition_authority",
    ]
    tissue_program["prohibited_structures"] = [
        "collapsed_zero",
        "encoded_airspace_or_lumen",
        "vessel",
        "necrosis",
        "immune_infiltrate",
        "normal_epithelium",
        "other_protected_profile_structure",
    ]
    tissue_program["front_contract"] = {
        "profile_mode": "tapered_lobe",
        "edge_depth_ratio": 0.12,
        "taper_fraction": 0.42,
        "lobe_count": 1,
        "noise_depth_ratio": 0.02,
        "maximum_band_px": 96,
        "maximum_depth_span_ratio": 4.0,
        "maximum_boundary_compactness": 14.0,
        "directional_sector_required": True,
        "maximum_selected_anchor_fraction": 0.5,
        "minimum_unselected_anchor_count": 1,
    }
    contract["cell_program"].update(
        actions=["retain", "remove_whole", "add"],
        allowed_cell_classes=[1],
        layout_programs=["boundary_aligned"],
        layout_program_by_primitive={
            "infiltrative-nest-cord-extension-v1": "boundary_aligned"
        },
        core_policy="replace_complete_instances_in_narrow_tissue_extension",
        halo_policy="no_cell_only_halo",
        halo_distance_px=[0, 12],
        cluster_size_range=[2, 6],
    )
    contract["coupling_contract"].update(
        compatibility_rule_ids=[
            "ignite.mask.synthetic_cord_connected",
            "ignite.mask.boundary_aligned_population_in_extension",
        ],
        allow_neoplastic_in_non_tumor_tissue=False,
        joint_area_mode="joint_footprint",
        tissue_floor_applies=True,
        cell_only_target_fraction=0.0,
    )
    contract["joint_gate_ids"] = [
        "joint_area",
        "tissue_floor",
        "cell_tissue_compatibility",
        "cell_zone_localization",
        "joint_provenance",
        "external_boundary_binding",
        "annotation_anchored_extension_geometry",
        "profile_fine_transition_authority",
        f"mechanism_postcondition:{contract['mechanism_id']}",
    ]
    contract["planner_policy"]["hard_constraint_checker_ids"] = [
        "external_boundary_binding",
        "annotation_anchored_extension_geometry",
        "profile_fine_transition_authority",
        "whole_instance_changes",
    ]
    contract["render_contract"] = {
        "required_findings": [
            "one connected annotation-anchored narrow synthetic Tumor extension",
            "complete class-1 population continuity inside the changed Tumor footprint",
        ],
        "veto_findings": [
            "remote island, side merge or protected-profile overlap",
            "histologic invasive-front, subtype or prognostic claim",
        ],
        "mask_guarantees": [
            f"only operational {host} converts to Tumor",
            "complete-instance nuclei edits only",
        ],
        "render_only_claims": [
            "histologic subtype identity",
            "desmoplastic reaction",
        ],
    }
    contract["counterexamples"] = [
        "remote Tumor island",
        "broad fill instead of one tapered extension",
        "edit of an encoded protected structure",
    ]


def _new_mechanism(
    writer: Writer,
    *,
    template_id: str,
    mechanism_id: str,
    domain_id: str,
    transform: Callable[[dict[str, Any]], None],
    skill_text: str,
    pathology_sources: list[str],
) -> dict[str, Any]:
    root = writer.root
    template_dir = root / MECHANISMS / template_id
    contract = copy.deepcopy(_contract(root, template_id))
    contract["mechanism_id"] = mechanism_id
    contract["pathology_domain_id"] = domain_id
    transform(contract)
    contract["joint_gate_ids"] = [
        item
        for item in contract["joint_gate_ids"]
        if not item.startswith("mechanism_postcondition:")
    ] + [f"mechanism_postcondition:{mechanism_id}"]
    base = MECHANISMS / mechanism_id
    writer.json(base / "references" / "joint_contract.json", contract)

    evidence = _load(template_dir / "references" / "evidence.json")
    evidence["skill_id"] = mechanism_id
    evidence["dataset_fact_policy"] = (
        "Dataset-label authority is supplied only by the independently composed "
        "annotation-profile skill."
    )
    for record in evidence["records"]:
        record["evidence_id"] = record["evidence_id"].replace(template_id, mechanism_id)
        if record["authority_category"] == "pathology_fact":
            record["source_ids"] = pathology_sources
        if record["authority_category"] == "engineering_proxy":
            record["claim_scope"] = sorted(
                {*record.get("claim_scope", []), "planner_policy"}
            )
    writer.json(base / "references" / "evidence.json", evidence)
    writer.text(base / "SKILL.md", skill_text.rstrip() + "\n")
    writer.text(
        base / "agents" / "openai.yaml",
        "interface:\n"
        f'  display_name: "{mechanism_id}"\n'
        '  short_description: "Shadow-only certified mask mechanism"\n'
        f'  default_prompt: "Use ${mechanism_id} only with certified mask candidates and required authority."\n',
    )
    writer.json(
        base / "references" / "counterexamples.json",
        {
            "mechanism_id": mechanism_id,
            "counterexamples": list(contract.get("counterexamples", ())),
        },
    )
    writer.json(
        base / "references" / "statistics.json",
        {
            "status": "uncalibrated",
            "production_allowed": False,
            "required_strata": [
                "dataset_revision",
                "patient_or_wsi",
                "pixel_size",
                "primitive",
                "mechanism",
            ],
            "metrics": [
                "final_mask_postcondition_pass_rate",
                "meaningful_effect_floor_pass_rate",
                "whole_instance_integrity",
                "frozen_generator_condition_response",
            ],
            "note": (
                "Shadow-only until profile-specific cohort calibration, frozen "
                "runtime validation, and independent pathology review are complete."
            ),
        },
    )
    return contract


def _retreat_transform(contract: dict[str, Any]) -> None:
    domain_id = contract["pathology_domain_id"]
    receiving_label = (
        "Other tissue"
        if domain_id == "oral-squamous-cell-carcinoma-v1"
        else "Stroma"
    )
    contract["summary"] = (
        "Retreat annotation-defined Tumor into the profile's operational "
        "non-tumor receiving label under explicit post-treatment context."
    )
    contract["recognition_contract"] = {
        "required_observations": [
            "explicit post-treatment semantic intent",
            "annotation-defined Tumor component and receiving boundary",
            "compiler-certified residual topology and cell capacity",
        ],
        "contraindications": [
            "fibrosis, tumor-bed, pCR or clinical-benefit claim",
            "protected profile label in the retreat support",
            "treatment context absent",
        ],
        "minimum_confidence": 0.9,
    }
    contract["representability_contract"]["required_auxiliary_structures"] = []
    contract["tissue_program"]["mode"] = (
        "operational_tumor_retreat_to_profile_receiver"
    )
    # Operational retreat is a mask-defined footprint contraction. It must
    # not inherit Breast's directional front selector from a schema template.
    contract["tissue_program"].pop("front_contract", None)
    contract["tissue_program"]["required_checker_ids"] = [
        "tissue_gate_binding",
        "profile_fine_transition_authority",
    ]
    contract["tissue_program"]["prohibited_structures"] = {
        "melanoma-v1": ["zero", "epidermis", "vessel", "necrosis"],
        "oral-squamous-cell-carcinoma-v1": ["fragmented_non_tissue"],
    }.get(domain_id, ["zero", "protected_profile_label"])
    contract["joint_gate_ids"] = [
        item
        for item in contract["joint_gate_ids"]
        if item != "bcss_operational_stroma_authority"
    ]
    contract["planner_policy"]["hard_constraint_checker_ids"] = [
        "profile_fine_transition_authority",
        "whole_instance_changes",
        "residual_fragmentation_topology",
    ]
    contract["cell_program"]["halo_policy"] = (
        "operational_receiver_compatible_non_neoplastic_population"
    )
    _add_retreat_primitives(contract)
    if domain_id == "oral-squamous-cell-carcinoma-v1":
        contract["supported_primitives"] = [
            item
            for item in contract["supported_primitives"]
            if item != "stroma-increase-v1"
        ]
        contract["tissue_program"]["primitive_label_contracts"].pop(
            "stroma-increase-v1", None
        )
        contract["cell_program"]["layout_program_by_primitive"].pop(
            "stroma-increase-v1", None
        )
    for value in contract["tissue_program"]["primitive_label_contracts"].values():
        value["target_labels"] = [receiving_label]
    contract["coupling_contract"]["compatibility_rule_ids"] = [
        "other_organ.retreat.explicit_post_treatment_context",
        "other_organ.retreat.profile_fine_transition_authority",
        "other_organ.retreat.complete_instance_turnover",
        "other_organ.retreat.residual_topology",
    ]
    contract["render_contract"] = {
        "required_findings": [
            "annotation-defined Tumor retreats only along a certified receiving boundary",
            "residual Tumor and all unrequested profile labels remain stable",
        ],
        "veto_findings": [
            "fibrosis, tumor-bed, pCR, response-percentage or clinical-benefit claim",
            "edit of a protected or unrequested profile label",
            "whole-lesion clearance unless separately authorized by a local ROI contract",
        ],
        "mask_guarantees": [
            f"changed tissue is Tumor to operational {receiving_label} only",
            "every changed nucleus is handled as a complete instance",
            "residual topology and the requested area floor are reconstructed from final masks",
        ],
        "render_only_claims": [
            "fibrosis",
            "tumor bed",
            "pCR",
            "response percentage",
            "clinical benefit",
        ],
    }
    contract["evidence_citations"] = {
        "prostate-adenocarcinoma-v1": [
            "https://www.kaggle.com/c/prostate-cancer-grade-assessment"
        ],
        "lung-carcinoma-v1": ["https://zenodo.org/records/17735903"],
        "melanoma-v1": [
            "https://puma.grand-challenge.org/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11837757/",
        ],
        "oral-squamous-cell-carcinoma-v1": [
            "https://sites.google.com/unibas.it/orca/home"
        ],
    }.get(domain_id, [])
    contract["counterexamples"] = [
        "inferring treatment context or receiving tissue identity from H&E",
        "calling operational label turnover fibrosis, tumor bed or clinical response",
        "accepting a final mask that violates residual topology or the requested floor",
    ]


def _clearance_transform(contract: dict[str, Any]) -> None:
    domain_id = contract["pathology_domain_id"]
    receiving_label = (
        "Other tissue"
        if domain_id == "oral-squamous-cell-carcinoma-v1"
        else "Stroma"
    )
    contract["summary"] = (
        "Clear annotation-defined Tumor only inside a digest-bound user ROI "
        "into the profile's operational non-tumor receiving label."
    )
    contract["recognition_contract"]["required_observations"] = [
        "explicit local-clearance request",
        "digest-bound user local_clearance_roi",
        "Tumor inside the ROI adjoining the operational receiving label",
    ]
    contract["recognition_contract"]["contraindications"] = [
        "missing, inferred, stale or digest-detached ROI",
        "protected or unrequested profile label inside the requested change",
        "whole-lesion, negative-margin, pCR or complete-response claim",
    ]
    contract["tissue_program"]["required_checker_ids"] = [
        "tissue_gate_binding",
        "profile_fine_transition_authority",
        "local_clearance_roi_binding",
    ]
    contract["joint_gate_ids"] = [
        item
        for item in contract["joint_gate_ids"]
        if item != "bcss_operational_stroma_authority"
    ]
    contract["joint_gate_ids"][-1] = (
        f"mechanism_postcondition:{contract['mechanism_id']}"
    )
    contract["planner_policy"]["hard_constraint_checker_ids"] = [
        "local_clearance_roi_binding",
        "profile_fine_transition_authority",
        "whole_instance_changes",
    ]
    contract["tissue_program"]["primitive_label_contracts"][
        "local-invasive-clearance-v1"
    ]["target_labels"] = [receiving_label]
    contract["tissue_program"]["prohibited_structures"] = {
        "prostate-adenocarcinoma-v1": [
            "zero",
            "benign_epithelium",
            "unrequested_pattern",
            "native_lumen",
        ],
        "lung-carcinoma-v1": [
            "zero",
            "necrosis",
            "immune",
            "normal_epithelium",
            "vessel",
            "other",
        ],
        "melanoma-v1": ["zero", "epidermis", "vessel", "necrosis"],
        "oral-squamous-cell-carcinoma-v1": ["fragmented_non_tissue"],
    }[domain_id]
    contract["coupling_contract"]["compatibility_rule_ids"] = [
        "other_organ.clearance.explicit_digest_bound_roi",
        "other_organ.clearance.profile_tumor_to_receiver_only",
        "other_organ.clearance.no_whole_lesion_claim",
    ]
    contract["render_contract"] = {
        "required_findings": [
            "annotation-defined Tumor is absent only inside the declared local ROI",
            "tissue and nuclei outside the ROI remain stable except complete-instance closure",
        ],
        "veto_findings": [
            "edit outside the declared ROI",
            "edit of a protected or unrequested profile label",
            "negative-margin, pCR or complete-response claim",
        ],
        "mask_guarantees": [
            f"ROI-bounded Tumor to operational {receiving_label} transition only",
            "changed-instance centers and changed tissue are reconstructed against the digest-bound ROI",
        ],
        "render_only_claims": [
            "negative margin",
            "pCR",
            "complete response",
            "clinical benefit",
        ],
    }
    contract["evidence_citations"] = {
        "prostate-adenocarcinoma-v1": [
            "https://www.kaggle.com/c/prostate-cancer-grade-assessment"
        ],
        "lung-carcinoma-v1": ["https://zenodo.org/records/17735903"],
        "melanoma-v1": ["https://puma.grand-challenge.org/"],
        "oral-squamous-cell-carcinoma-v1": [
            "https://sites.google.com/unibas.it/orca/home"
        ],
    }[domain_id]
    contract["counterexamples"] = [
        "auto-inferring a clearance ROI from H&E or the Tumor component",
        "editing Tumor outside the digest-bound local ROI",
        "calling local annotation turnover a negative margin, pCR or complete response",
    ]


def _immune_transform(contract: dict[str, Any]) -> None:
    contract["summary"] = (
        "Turn over an existing annotation-defined Stroma/Immune-infiltrate "
        "interface with generic class-2/class-3 populations only."
    )
    contract["recognition_contract"]["required_observations"] = [
        "explicit Immune-infiltrate label",
        "existing Stroma/Immune-infiltrate interface",
        "complete class-2 inflammatory and class-3 connective references",
    ]
    contract["recognition_contract"]["contraindications"] = [
        "remote de-novo Immune-infiltrate island",
        "Tumor, Necrosis, Normal epithelium, Vessel, Other or zero in the edit band",
        "immune subtype, PD-L1, TIL-score, response, prognosis or benefit claim",
    ]
    contract["tissue_program"]["required_checker_ids"] = [
        "tissue_gate_binding",
        "profile_fine_transition_authority",
    ]
    contract["joint_gate_ids"] = [
        item
        for item in contract["joint_gate_ids"]
        if item != "bcss_operational_stroma_authority"
    ]
    contract["joint_gate_ids"][-1] = (
        f"mechanism_postcondition:{contract['mechanism_id']}"
    )
    contract["planner_policy"]["hard_constraint_checker_ids"] = [
        "profile_fine_transition_authority",
        "whole_instance_changes",
    ]
    contract["tissue_program"]["prohibited_structures"] = [
        "tumor",
        "necrosis",
        "normal_epithelium",
        "vessel",
        "other",
        "zero",
    ]
    contract["coupling_contract"]["compatibility_rule_ids"] = [
        "ignite.immune.existing_stroma_immune_interface",
        "ignite.immune.direction_specific_whole_instance_turnover",
        "ignite.immune.no_subtype_or_outcome_claim",
    ]
    contract["render_contract"] = {
        "required_findings": [
            "only the existing IGNITE Stroma/Immune-infiltrate interface turns over",
            "the direction-specific generic class-2/class-3 population is regenerated",
        ],
        "veto_findings": [
            "remote Immune-infiltrate island",
            "change to Tumor, Necrosis, Normal epithelium, Vessel, Other or zero",
            "immune subtype, PD-L1, TIL-score, response, prognosis or benefit claim",
        ],
        "mask_guarantees": [
            "fine-2/fine-4 tissue transition only",
            "complete class-2/class-3 instance turnover only",
        ],
        "render_only_claims": [
            "immune subtype",
            "PD-L1",
            "TIL score",
            "response",
            "prognosis",
            "clinical benefit",
        ],
    }
    contract["evidence_citations"] = [
        "https://zenodo.org/records/17735903"
    ]
    contract["counterexamples"] = [
        "creating a remote Immune-infiltrate island",
        "calling generic class-2 nuclei a specific immune subtype",
        "claiming PD-L1, TIL score, response, prognosis or benefit",
    ]


def _oral_cord_transform(contract: dict[str, Any]) -> None:
    contract["summary"] = (
        "Create one narrow connected ORCA Carcinoma extension from a certified "
        "external Tumor/Other-tissue interface; no invasive-pattern claim."
    )
    contract["recognition_contract"] = {
        "required_observations": [
            "explicit ORCA Tumor component and exterior boundary",
            "Other-tissue receiving corridor with exact zero exclusion",
            "compiler-certified tapered projection capacity",
        ],
        "contraindications": [
            "fragmented non-tissue in the projection corridor",
            "remote island or side merge",
            "WPOI, budding grade or invasive-front diagnosis requested",
        ],
        "minimum_confidence": 0.9,
    }
    contract["tissue_program"]["mode"] = (
        "orca_annotation_anchored_narrow_connected_extension"
    )
    contract["tissue_program"]["primitive_label_contracts"] = {
        "infiltrative-nest-cord-extension-v1": {
            "source_labels": ["Other tissue"],
            "target_labels": ["Tumor"],
        }
    }
    contract["tissue_program"]["required_checker_ids"] = [
        "tissue_gate_binding",
        "external_boundary_binding",
        "annotation_anchored_extension_geometry",
        "orca_fragment_protection",
    ]
    contract["tissue_program"]["prohibited_structures"] = [
        "fragmented_non_tissue",
        "surface_space",
    ]
    contract["joint_gate_ids"] = [
        item
        for item in contract["joint_gate_ids"]
        if item not in {"bcss_operational_stroma_authority", "profile_fine_transition_authority"}
    ]
    contract["joint_gate_ids"] = _ordered_union(
        contract["joint_gate_ids"], ["orca_fragment_protection"]
    )
    contract["joint_gate_ids"][-2:] = [
        "orca_fragment_protection",
        f"mechanism_postcondition:{contract['mechanism_id']}",
    ]
    contract["planner_policy"]["hard_constraint_checker_ids"] = [
        "external_boundary_binding",
        "annotation_anchored_extension_geometry",
        "orca_fragment_protection",
        "whole_instance_changes",
    ]
    contract["coupling_contract"]["compatibility_rule_ids"] = [
        "orca.mask.narrow_extension_connected",
        "orca.mask.boundary_aligned_population_in_narrow_extension",
        "orca.mask.zero_preserved",
    ]
    contract["evidence_citations"] = [
        "https://sites.google.com/unibas.it/orca/home"
    ]
    contract["counterexamples"] = [
        "detached fine-1 island",
        "broad semicircular fill instead of one narrow extension",
        "change to ORCA zero",
        "diagnostic WPOI, budding, PNI, LVI or prognostic claim",
    ]


def _install_primitives(writer: Writer) -> None:
    # The same non-diagnostic peritumoral layouts are valid against an
    # explicitly heterogeneous non-tumor receiver in ORCA.  This does not
    # reclassify Other tissue as Stroma; it merely authorizes footprint
    # containment in the profile's named host class.
    for primitive_id in (
        "peritumoral-neoplastic-scatter-increase-v1",
        "peritumoral-small-cluster-increase-v1",
    ):
        relative = (
            PRIMITIVES
            / primitive_id
            / "references"
            / "primitive_contract.json"
        )
        contract = _load(writer.root / relative)
        contract["host_tissue_labels"] = _ordered_union(
            contract["host_tissue_labels"], ["Other tissue"]
        )
        writer.json(relative, contract)

    cellularity_path = (
        PRIMITIVES
        / "cellularity-increase-v1"
        / "references"
        / "primitive_contract.json"
    )
    cellularity = _load(writer.root / cellularity_path)
    cellularity["cell_effect_contract"] = {
        "minimum_delta_count": 12,
        "minimum_delta_count_by_pathology_domain": {
            "prostate-adenocarcinoma-v1": 10,
        },
        "minimum_span_cell_diameters": 6.0,
        "minimum_foci": 4,
    }
    writer.json(cellularity_path, cellularity)
    cellularity_evidence_path = (
        PRIMITIVES
        / "cellularity-increase-v1"
        / "references"
        / "evidence.json"
    )
    cellularity_evidence = _load(writer.root / cellularity_evidence_path)
    cellularity_evidence["records"][0]["claim_scope"] = _ordered_union(
        cellularity_evidence["records"][0]["claim_scope"],
        ["cell_effect_contract"],
    )
    writer.json(cellularity_evidence_path, cellularity_evidence)
    writer.text(
        PRIMITIVES / "cellularity-increase-v1" / "SKILL.md",
        "---\n"
        "name: cellularity-increase-v1\n"
        "description: Compile a local cell-only increase in total cellularity without changing tissue labels. Use only when a reviewed mechanism and population profile define a legal zone, count increment, and cell composition.\n"
        "---\n\n"
        "# Cellularity increase\n\n"
        "Preserve tissue, resolve a count/extent budget, retain existing complete "
        "instances, and add a profile-compatible mixed population. The effect "
        "must add at least 12 complete instances across at least four foci and "
        "span six local cell diameters; otherwise abstain as packing-infeasible. "
        "Do not silently convert this primitive into neoplastic infiltration. "
        "Read [primitive_contract.json](references/primitive_contract.json).\n",
    )

    cluster_path = (
        PRIMITIVES
        / "peritumoral-small-cluster-increase-v1"
        / "references"
        / "primitive_contract.json"
    )
    cluster = _load(writer.root / cluster_path)
    cluster["summary"] = cluster["summary"].replace(
        "localized invasive-front hotspot", "localized peritumoral hotspot"
    )
    writer.json(cluster_path, cluster)
    writer.text(
        PRIMITIVES / "peritumoral-small-cluster-increase-v1" / "SKILL.md",
        "---\n"
        "name: peritumoral-small-cluster-increase-v1\n"
        "description: Add a localized hotspot of multiple tight 2--4-cell neoplastic foci to a certified outer Tumor--host annulus without claiming diagnostic tumor budding.\n"
        "---\n\n"
        "# Peritumoral Small-Cluster Increase\n\n"
        "Use this primitive for a cell-only set of complete class-1 foci adjacent to an annotated Tumor component.\n\n"
        "- Preserve tissue labels pixel-exactly.\n"
        "- Bind a certified external Tumor--host interface and outer annulus.\n"
        "- Select one finite peritumoral neighborhood rather than the full annulus.\n"
        "- Generate multiple tight foci containing two to four complete nuclei each.\n"
        "- Require visible within-focus adjacency and clear between-focus separation.\n"
        "- Keep all foci near the main Tumor component; reject solid bridges and remote deposits.\n"
        "- Describe the result as synthetic peritumoral small-cluster morphology, not a tumor-budding diagnosis or score.\n\n"
        "Read `references/primitive_contract.json` and `references/evidence.json` before execution.\n",
    )

    class_decrease_path = (
        PRIMITIVES
        / "cell-type-abundance-decrease-v1"
        / "references"
        / "primitive_contract.json"
    )
    class_decrease = _load(writer.root / class_decrease_path)
    class_decrease["cell_effect_contract"].setdefault(
        "minimum_delta_count_by_pathology_domain", {}
    ).update(
        {
            # ORCA class 2 is a generic inflammatory observation.  Ten
            # complete instances makes a focal density decrease reviewable
            # without converting it into near-total immune clearance.
            "oral-squamous-cell-carcinoma-v1": 10,
            "melanoma-v1": 10,
        }
    )
    writer.json(class_decrease_path, class_decrease)

    for direction in ("increase", "decrease"):
        primitive_id = f"generic-inflammatory-cell-abundance-{direction}-v1"
        source_id = f"neoplastic-cell-abundance-{direction}-v1"
        source_dir = writer.root / PRIMITIVES / source_id
        contract = _load(source_dir / "references" / "primitive_contract.json")
        contract["primitive_id"] = primitive_id
        contract["summary"] = (
            f"{direction.title()} only observable generic class-2 inflammatory "
            "nuclei in an existing mask-defined tissue component without "
            "creating an immune tissue label or subtype claim."
        )
        contract["host_tissue_labels"] = [
            "Tumor",
            "Stroma",
            "Other tissue",
            "Normal epithelium",
        ]
        contract["target_cell_classes"] = [2]
        if direction == "decrease":
            contract["cell_effect_contract"].setdefault(
                "minimum_delta_count_by_pathology_domain", {}
            ).update(
                {
                    "lung-carcinoma-v1": 12,
                    "oral-squamous-cell-carcinoma-v1": 10,
                    "melanoma-v1": 10,
                }
            )
        base = PRIMITIVES / primitive_id
        writer.json(base / "references" / "primitive_contract.json", contract)
        evidence = _load(source_dir / "references" / "evidence.json")
        evidence["skill_id"] = primitive_id
        for record in evidence["records"]:
            record["evidence_id"] = record["evidence_id"].replace(
                source_id, primitive_id
            )
        evidence["pathology_fact_policy"] = (
            "Class-2 means only the configured generic inflammatory observation; "
            "no immune subtype, TIL score, response or prognosis is authorized."
        )
        writer.json(base / "references" / "evidence.json", evidence)
        writer.text(
            base / "SKILL.md",
            "---\n"
            f"name: {primitive_id}\n"
            "description: Change generic inflammatory class-2 abundance in one certified tissue component without changing tissue semantics.\n"
            "---\n\n"
            f"# Generic inflammatory-cell abundance {direction}\n\n"
            "Read `references/primitive_contract.json`. Operate only on complete "
            "class-2 instances in a compiler-certified component. Do not create "
            "an immune tissue region or claim a subtype, TIL score, response, "
            "prognosis or clinical benefit.\n",
        )
        writer.text(
            base / "agents" / "openai.yaml",
            "interface:\n"
            f'  display_name: "Generic inflammatory-cell abundance {direction}"\n'
            '  short_description: "Change generic class-2 nuclei only"\n'
            f'  default_prompt: "Use ${primitive_id} only for a certified mask-defined component."\n',
        )


def _melanoma_small_focus_transform(contract: dict[str, Any]) -> None:
    """Keep attempted stromal focus programs explicit for fail-closed audit."""

    scatter_id = "peritumoral-neoplastic-scatter-increase-v1"
    cluster_id = "peritumoral-small-cluster-increase-v1"
    contract["supported_primitives"] = [scatter_id, cluster_id]
    contract["summary"] = (
        "Add multiple separated complete class-1 small clusters in a certified "
        "Tumor/Stroma outer annulus; this is not a microsatellite diagnosis."
    )
    contract["recognition_contract"] = {
        "required_observations": [
            "explicit Tumor/Stroma external mask boundary",
            "compiler-certified bounded stromal annulus",
            "source-matched complete class-1 reference nuclei",
        ],
        "contraindications": [
            "necrosis, vessel, epidermis or zero in a proposed footprint",
            "remote focus beyond the bounded annulus",
            "microsatellite or metastatic diagnostic claim",
        ],
        "minimum_confidence": 0.9,
    }
    contract["representability_contract"]["required_auxiliary_structures"] = []
    contract["tissue_program"]["primitive_label_contracts"] = {
        primitive_id: {
            "source_labels": ["Stroma"],
            "target_labels": ["Stroma"],
        }
        for primitive_id in contract["supported_primitives"]
    }
    contract["cell_program"]["layout_program_by_primitive"] = {
        scatter_id: "single",
        cluster_id: "small_cluster",
    }
    contract["cell_program"]["layout_programs"] = ["single", "small_cluster"]
    contract["cell_program"]["cluster_size_range"] = [2, 4]
    contract["cell_program"]["required_checker_ids"] = [
        item
        for item in contract["cell_program"]["required_checker_ids"]
        if item != "puma_epidermal_junction_binding"
    ]
    contract["planner_policy"]["hard_constraint_checker_ids"] = [
        item
        for item in contract["planner_policy"]["hard_constraint_checker_ids"]
        if item != "puma_epidermal_junction_binding"
    ]
    contract["tissue_program"]["required_checker_ids"] = [
        item
        for item in contract["tissue_program"]["required_checker_ids"]
        if item != "puma_epidermal_junction_binding"
    ]
    contract["joint_gate_ids"] = [
        item
        for item in contract["joint_gate_ids"]
        if item != "puma_epidermal_junction_binding"
    ]
    contract["joint_gate_ids"][-1] = (
        f"mechanism_postcondition:{contract['mechanism_id']}"
    )
    contract["render_contract"] = {
        "required_findings": [
            "multiple separated 2-4-cell synthetic foci in the bounded stromal annulus",
            "source Tumor footprint remains pixel-exact",
        ],
        "veto_findings": [
            "bridge to the primary Tumor component",
            "focus outside the certified annulus",
            "microsatellite, metastasis, staging or prognosis claim",
        ],
        "mask_guarantees": [
            "tissue mask is pixel-exact",
            "all added class-1 footprints are complete and annulus-contained",
        ],
        "render_only_claims": [
            "microsatellite",
            "metastasis",
            "staging",
            "prognosis",
        ],
    }
    contract["evidence_citations"] = [
        "https://puma.grand-challenge.org/",
        "https://documents.cap.org/documents/New-Cancer-Protocols-March-2025/Skin.Inv_Melanoma.Bx_1.1.0.0.REL.CAPCP.pdf",
    ]
    contract["counterexamples"] = [
        "calling generated small foci diagnostic microsatellites",
        "placing a focus in epidermis, vessel, necrosis or zero",
        "accepting a remote or primary-connected focus",
    ]


def _prostate_pattern5_scatter_transform(contract: dict[str, Any]) -> None:
    """Make PANDA scatter a fine-10-bound, lumen-protected cell program."""

    primitive_id = "peritumoral-neoplastic-scatter-increase-v1"
    contract["supported_primitives"] = [primitive_id]
    contract["summary"] = (
        "Add sparse complete class-1 nuclei outside an explicit fine-10 "
        "Pattern-5/Stroma boundary while preserving PANDA tissue, pattern "
        "labels and native enclosed spaces."
    )
    contract["recognition_contract"] = {
        "required_observations": [
            "explicit fine-10 Pattern-5 component",
            "selected fine-10 to fine-2 Stroma boundary anchor",
            "digest-bound deterministic profile-produced native pattern/lumen protection map",
            "source-matched complete class-1 reference nuclei",
        ],
        "contraindications": [
            "selected anchor contacts only fine-8 or fine-9 Tumor",
            "native lumen or enclosed pattern space in a proposed footprint",
            "remote focus, solid bridge or diagnostic grade/invasion claim",
        ],
        "minimum_confidence": 0.9,
    }
    contract["representability_contract"] = {
        "status": "conditionally_supported",
        "required_cell_classes": [1],
        "required_auxiliary_structures": [
            "native_pattern_and_lumen_map"
        ],
        "protected_auxiliary_structures": [
            "native_pattern_and_lumen_map"
        ],
        "allow_semantic_instance_fallback": False,
        "failure_action": "abstain_case",
    }
    contract["tissue_program"]["mode"] = (
        "preserve_tissue_at_fine10_pattern5_external_annulus"
    )
    contract["tissue_program"]["primitive_label_contracts"] = {
        primitive_id: {
            "source_labels": ["Stroma"],
            "target_labels": ["Stroma"],
        }
    }
    contract["tissue_program"]["allowed_tools"] = ["preserve_tissue"]
    contract["tissue_program"]["prohibited_structures"] = [
        "gland_lumen",
        "cribriform_internal_space",
        "glomeruloid_unit",
        "vessel",
        "nerve",
        "zero",
    ]
    contract["tissue_program"]["required_checker_ids"] = [
        "tissue_gate_binding",
        "external_boundary_binding",
        "native_structure_preserved",
        "fine_pattern_preserved",
        "panda_pattern5_scatter_binding",
    ]
    contract["cell_program"]["actions"] = ["retain", "add"]
    contract["cell_program"]["allowed_cell_classes"] = [1]
    contract["cell_program"]["layout_programs"] = ["single"]
    contract["cell_program"]["layout_program_by_primitive"] = {
        primitive_id: "single"
    }
    contract["cell_program"]["halo_distance_px"] = [4, 48]
    contract["cell_program"]["cluster_size_range"] = [1, 1]
    contract["cell_program"]["halo_policy"] = (
        "add_complete_class1_outside_selected_fine10_pattern5_boundary"
    )
    contract["cell_program"]["required_checker_ids"] = _ordered_union(
        contract["cell_program"]["required_checker_ids"],
        [
            "native_structure_preserved",
            "fine_pattern_preserved",
            "panda_pattern5_scatter_binding",
        ],
    )
    contract["coupling_contract"].update(
        allow_neoplastic_in_non_tumor_tissue=True,
        joint_area_mode="cell_count_extent",
        tissue_floor_applies=False,
        cell_only_target_fraction=1,
    )
    contract["coupling_contract"]["compatibility_rule_ids"] = [
        "panda.mask.fine10_pattern5_outer_annulus",
        "panda.mask.native_lumen_preserved",
        "panda.mask.sparse_single_class1",
    ]
    contract["planner_policy"]["prohibited_observation_sources"] = [
        "source_he_for_execution",
        "unannotated_histology_inference",
        "gleason_grade_inference",
        "histologic_invasion_inference",
    ]
    contract["planner_policy"]["clarification_triggers"] = [
        "the instruction could mean tissue-level Pattern-5 expansion rather than cell-only scatter",
        "the user requests diagnostic Gleason grade or invasion authority",
    ]
    contract["planner_policy"]["hard_constraint_checker_ids"] = [
        "external_boundary_binding",
        "native_structure_preserved",
        "fine_pattern_preserved",
        "panda_pattern5_scatter_binding",
        "peritumoral_annulus",
        "peritumoral_scatter_separation",
        "no_remote_neoplastic_focus",
        "no_solid_neoplastic_bridge",
    ]
    contract["joint_gate_ids"] = [
        "cell_tissue_compatibility",
        "cell_zone_localization",
        "joint_provenance",
        "external_boundary_binding",
        "native_structure_preserved",
        "fine_pattern_preserved",
        "panda_pattern5_scatter_binding",
        "peritumoral_annulus",
        "peritumoral_scatter_separation",
        "no_remote_neoplastic_focus",
        "no_solid_neoplastic_bridge",
        f"mechanism_postcondition:{contract['mechanism_id']}",
    ]
    contract["render_contract"] = {
        "required_findings": [
            "sparse complete class-1 instances outside the selected fine-10 boundary",
            "PANDA fine labels and native enclosed spaces remain unchanged",
        ],
        "veto_findings": [
            "scatter anchored only to fine-8 or fine-9",
            "nucleus in a native enclosed space",
            "remote deposit or solid bridge to the Pattern-5 component",
        ],
        "mask_guarantees": [
            "tissue and PANDA fine labels are pixel-exact",
            "complete single class-1 instances remain in the certified fine-10 outer annulus",
        ],
        "render_only_claims": [
            "Gleason grade change",
            "histologic invasion",
            "extraprostatic extension",
            "prognosis",
        ],
    }
    contract["counterexamples"] = [
        "anchoring scatter to a fine-8 or fine-9-only boundary",
        "placing cells in a protected native lumen",
        "calling synthetic peripheral scatter diagnostic invasion or grade progression",
    ]
    contract["evidence_citations"] = [
        "https://panda.grand-challenge.org/Data/",
        "https://doi.org/10.1038/s41591-021-01620-2",
    ]


def _update_profile(
    writer: Writer,
    *,
    profile_id: str,
    conditional: list[str],
    sources: dict[str, list[int]],
    targets: dict[str, list[int]],
    operational_stroma_ids: list[int] | None,
    required: dict[str, list[int]] | None = None,
) -> None:
    contract_path = PROFILES / profile_id / "references" / "joint_contract.json"
    contract = _load(writer.root / contract_path)
    contract["conditional_mechanisms"] = _ordered_union(
        contract.get("conditional_mechanisms", []), conditional
    )
    for obsolete in (
        "source_fine_ids_by_mechanism",
        "target_fine_ids_by_mechanism",
        "edit_map_fine_ids_by_mechanism",
        "visual_veto_fine_ids_by_mechanism",
        "stroma_label_proves_fibrosis",
    ):
        contract.pop(obsolete, None)
    contract.setdefault("mechanism_editable_source_fine_ids", {}).update(sources)
    contract.setdefault("mechanism_editable_target_fine_ids", {}).update(targets)
    if required:
        contract.setdefault("mechanism_required_fine_ids", {}).update(required)
    elif contract.get("mechanism_required_fine_ids") == {}:
        contract.pop("mechanism_required_fine_ids")
    contract.setdefault("protected_fine_ids", [0])
    if operational_stroma_ids is not None:
        contract["operational_stroma_fine_ids"] = operational_stroma_ids
        contract["operational_stroma_policy"] = (
            "The configured label is an operational receiving class only and "
            "does not prove fibrosis, tumor bed, response or clinical benefit."
        )
        contract["fibrosis_claim_authorized"] = False
    if not isinstance(contract.get("visual_veto_requirements"), list):
        contract["visual_veto_requirements"] = []
    writer.json(contract_path, contract)

    evidence_path = PROFILES / profile_id / "references" / "evidence.json"
    evidence = _load(writer.root / evidence_path)
    obsolete_claims = {
        "source_fine_ids_by_mechanism",
        "target_fine_ids_by_mechanism",
        "edit_map_fine_ids_by_mechanism",
        "visual_veto_fine_ids_by_mechanism",
        "stroma_label_proves_fibrosis",
    }
    for record in evidence["records"]:
        record["claim_scope"] = [
            field
            for field in record.get("claim_scope", [])
            if field not in obsolete_claims
        ]
        if (
            operational_stroma_ids is not None
            and record["authority_category"] == "dataset_fact"
        ):
            record["claim_scope"] = sorted(
                {*record.get("claim_scope", []), "fibrosis_claim_authorized"}
            )
        if record["authority_category"] == "engineering_proxy":
            advanced_claims = {
                "mechanism_editable_source_fine_ids",
                "mechanism_editable_target_fine_ids",
                "protected_fine_ids",
                "visual_veto_requirements",
            }
            if operational_stroma_ids is not None:
                advanced_claims.update(
                    {
                        "operational_stroma_fine_ids",
                        "operational_stroma_policy",
                    }
                )
            record["claim_scope"] = sorted(
                {
                    *record.get("claim_scope", []),
                    *advanced_claims,
                }
            )
    writer.json(evidence_path, evidence)


def _assert_clean_non_breast_catalog(root: Path) -> None:
    """Fail generator checks on inherited Breast or ORCA semantic authority."""

    forbidden_cross_organ = (
        "breast",
        "bcss",
        "dcis",
        "angioinvasion",
        "benign_duct",
        "cap breast",
        "breast.mask",
    )
    for path in sorted(
        (root / MECHANISMS).glob("*/references/joint_contract.json")
    ):
        payload = _load(path)
        if payload.get("pathology_domain_id") == "breast-invasive-carcinoma-v1":
            continue
        text = json.dumps(payload, ensure_ascii=False).lower()
        contaminated = sorted(
            token for token in forbidden_cross_organ if token in text
        )
        if contaminated:
            raise ValueError(
                f"non-Breast mechanism contamination in {path}: {contaminated}"
            )
        if str(payload.get("mechanism_id", "")).endswith(
            "operational-tumor-retreat"
        ):
            halo_policy = (payload.get("cell_program") or {}).get("halo_policy")
            if halo_policy != (
                "operational_receiver_compatible_non_neoplastic_population"
            ):
                raise ValueError(
                    f"operational retreat has untyped receiver population in {path}"
                )
            positive_surfaces = {
                "summary": payload.get("summary"),
                "required_observations": (
                    payload.get("recognition_contract") or {}
                ).get("required_observations"),
                "tissue_program": payload.get("tissue_program"),
                "cell_program": payload.get("cell_program"),
                "required_findings": (
                    payload.get("render_contract") or {}
                ).get("required_findings"),
                "mask_guarantees": (
                    payload.get("render_contract") or {}
                ).get("mask_guarantees"),
            }
            positive_text = json.dumps(
                positive_surfaces, ensure_ascii=False
            ).lower()
            overclaims = sorted(
                token for token in ("tumor bed", "tumor-bed", "fibros")
                if token in positive_text
            )
            if overclaims:
                raise ValueError(
                    f"operational retreat execution overclaim in {path}: {overclaims}"
                )
        if payload.get("pathology_domain_id") == (
            "oral-squamous-cell-carcinoma-v1"
        ):
            execution_surfaces = {
                "summary": payload.get("summary"),
                "required_observations": (
                    payload.get("recognition_contract") or {}
                ).get("required_observations"),
                "representability_contract": payload.get(
                    "representability_contract"
                ),
                "tissue_program": payload.get("tissue_program"),
                "cell_program": payload.get("cell_program"),
                "coupling_contract": payload.get("coupling_contract"),
                "required_findings": (
                    payload.get("render_contract") or {}
                ).get("required_findings"),
                "mask_guarantees": (
                    payload.get("render_contract") or {}
                ).get("mask_guarantees"),
            }
            authority_text = json.dumps(
                execution_surfaces, ensure_ascii=False
            ).lower()
            overclaims = sorted(
                token
                for token in ("stroma", "fibros", "immune tissue")
                if token in authority_text
            )
            if overclaims:
                raise ValueError(
                    f"ORCA execution semantic overclaim in {path}: {overclaims}"
                )


def refine(root: Path, *, check: bool) -> list[Path]:
    writer = Writer(root, check=check)
    _install_primitives(writer)
    fragmentation_path = (
        CATALOG
        / "edit-primitive"
        / "residual-tumor-fragmentation-v1"
        / "references"
        / "primitive_contract.json"
    )
    fragmentation = _load(root / fragmentation_path)
    fragmentation["version"] = "1.1.15-draft"
    fragmentation_topology = fragmentation["tissue_topology_contract"]
    fragmentation_topology["minimum_residual_components"] = 2
    fragmentation_topology["maximum_residual_components"] = 6
    writer.json(fragmentation_path, fragmentation)
    execution_scope_path = CATALOG / "execution-scope-v1.json"
    execution_scope = _load(root / execution_scope_path)
    execution_scope["executable_primitives"] = _ordered_union(
        execution_scope["executable_primitives"],
        [
            "generic-inflammatory-cell-abundance-increase-v1",
            "generic-inflammatory-cell-abundance-decrease-v1",
        ],
    )
    closed_pairs = execution_scope.setdefault("closed_pairs", {})
    # Re-open the temporary empirical closures after replacing the old
    # pixel-area-dominant meta ranking with mask-component-aware ranking.
    for pair_id in (
        "oral-scc-local-population-modulation::cell-type-abundance-decrease-v1",
        "oral-scc-local-population-modulation::cellularity-increase-v1",
        "oral-scc-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1",
        "melanoma-local-population-modulation::cell-type-abundance-decrease-v1",
        "melanoma-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1",
        "melanoma-local-population-modulation::neoplastic-cell-abundance-decrease-v1",
    ):
        closed_pairs.pop(pair_id, None)
    closed_pairs.update(
        {
            "lung-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1": (
                "None of 12 top IGNITE cases selected after corrected mask-component ranking satisfied the "
                "generic inflammatory complete-instance depletion contract. The pair cannot provide robust "
                "five-case execution, while the separate cell-type decrease remains executable."
            ),
            "lung-local-population-modulation::neoplastic-cell-abundance-decrease-v1": (
                "None of six top IGNITE cases selected after corrected mask-component ranking provided a legal "
                "tumor zone satisfying complete neoplastic-instance depletion, the local gradient and residual "
                "population floors. The primitive cannot provide robust five-case execution."
            ),
            "lung-local-population-modulation::neoplastic-cell-abundance-increase-v1": (
                "The first three top IGNITE cases all exceeded the 240-second execution limit without "
                "producing a candidate summary. The mature executor cannot provide robust five-case "
                "neoplastic addition under the current runtime contract."
            ),
            "lung-local-population-modulation::peritumoral-neoplastic-scatter-increase-v1": (
                "None of seven top IGNITE Tumor-Stroma interface cases could pack the required ten separated "
                "complete neoplastic foci; capacity was typically one or two. Lowering the topology floor "
                "would make the edit too small to be meaningful."
            ),
            "lung-local-population-modulation::peritumoral-small-cluster-increase-v1": (
                "None of six top IGNITE Tumor-Stroma interface cases could pack the required eight separated "
                "complete cluster units; observed capacity was one to three. Lowering the topology floor "
                "would make the edit too small to be meaningful."
            ),
            "oral-scc-local-population-modulation::cell-type-abundance-decrease-v1": (
                "After mask-component-aware reranking, only the one preserved success remained and 11 further "
                "top ORCA cases failed the complete-instance depletion, three-band gradient or residual floors. "
                "The primitive cannot provide robust five-case execution."
            ),
            "oral-scc-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1": (
                "ORCA binds this intent to the same inflammatory class and complete-instance depletion program "
                "as cell-type decrease, which remained 1-of-12 after corrected mask-component ranking. The "
                "primitive cannot provide robust five-case execution."
            ),
            "melanoma-local-population-modulation::cell-type-abundance-decrease-v1": (
                "After mask-component-aware reranking, only the one preserved success remained and 13 further "
                "top PUMA cases failed the complete-instance depletion, three-band gradient or residual floors. "
                "The primitive cannot provide robust five-case execution."
            ),
            "melanoma-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1": (
                "PUMA binds this intent to the same inflammatory class and complete-instance depletion program "
                "as cell-type decrease, which remained 1-of-14 after corrected mask-component ranking. The "
                "primitive cannot provide robust five-case execution."
            ),
            "melanoma-local-population-modulation::neoplastic-cell-abundance-decrease-v1": (
                "None of 12 top PUMA cases selected after corrected mask-component ranking provided a legal "
                "tumor zone satisfying complete neoplastic-instance depletion, the local gradient and residual "
                "population floors. The primitive cannot provide robust five-case execution."
            ),
        }
    )
    # Runtime failures are implementation defects, not annotation closures.
    # Keep these pairs visible to the compiler while their executors are
    # repaired and validated; only missing mask authority may remain closed.
    implementation_pairs = {
        "lung-generic-immune-compartment-turnover::generic-immune-infiltrate-decrease-v1",
        "lung-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1",
        "lung-local-population-modulation::neoplastic-cell-abundance-decrease-v1",
        "lung-local-population-modulation::neoplastic-cell-abundance-increase-v1",
        "lung-local-population-modulation::peritumoral-neoplastic-scatter-increase-v1",
        "lung-local-population-modulation::peritumoral-small-cluster-increase-v1",
        "oral-scc-local-population-modulation::cell-type-abundance-decrease-v1",
        "oral-scc-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1",
        "melanoma-local-population-modulation::cell-type-abundance-decrease-v1",
        "melanoma-local-population-modulation::generic-inflammatory-cell-abundance-decrease-v1",
        "melanoma-local-population-modulation::neoplastic-cell-abundance-decrease-v1",
    }
    for pair_id in implementation_pairs:
        closed_pairs.pop(pair_id, None)

    closed_mechanisms = execution_scope.setdefault("closed_mechanisms", {})
    closed_mechanisms.update(
        {
            "lung-acinar-papillary-growth": (
                "IGNITE does not provide a digest-bound lumen or fibrovascular-core map. "
                "Acinar/papillary identity cannot be inferred from H&E because execution planning is mask-only."
            ),
            "lung-lepidic-growth": (
                "IGNITE does not provide a digest-bound alveolar framework map. "
                "Lepidic growth cannot be executed without risking airspace filling, and H&E inference is prohibited."
            ),
            "lung-local-tumor-clearance": (
                "The cross-validation annotation bundle has no user-supplied, digest-bound "
                "local_clearance_roi; automatic ROI invention would not be a mask-authorized local clearance."
            ),
            "lung-stas-airspace-spread": (
                "STAS requires explicit alveolar/airspace authority. The current IGNITE bundle has no "
                "digest-bound airspace map, so mask-only execution must fail closed."
            ),
            "lung-generic-immune-compartment-turnover": (
                "Immune-compartment decrease produced three 300-second timeouts and one deterministic rejection, "
                "while the first three ranked increase cases had zero nuclei-safe interface capacity. The shared "
                "tissue-compartment mechanism cannot provide robust five-case execution."
            ),
            "lung-intratumoral-necrosis-turnover": (
                "All six mask-ranked appearance cases and all three mask-ranked resolution cases had zero "
                "nuclei-safe tissue capacity at the 5% visible-area floor. The shared turnover mechanism "
                "cannot execute without cutting complete annotated nuclei under the current authority."
            ),
            "lung-solid-squamous-growth": (
                "Corrected mask ranking and 12 top IGNITE retries still produced no executable cohesive "
                "interface at a 4% visible-area floor; failures were deterministic replanning stalls or zero "
                "nuclei-safe capacity. Both shared growth primitives therefore remain closed."
            ),
            "lung-stromal-invasive-front": (
                "The first five corrected-rank IGNITE cases all failed native instance preflight because the "
                "hybrid partition could not authorize complete-nucleus removal; one also lacked the required "
                "native reference class. Cord extension is unavailable under the current authority."
            ),
            "lung-operational-tumor-retreat": (
                "The first three top IGNITE tumor-interface cases all failed native instance preflight because "
                "the hybrid partition could not authorize complete-nucleus removal. The three shared retreat "
                "programs are unavailable under the current authority."
            ),
            "melanoma-discohesive-junctional": (
                "PUMA epidermis labels do not identify a junctional melanoma component. Junctional/pagetoid "
                "spread cannot be inferred from H&E by the execution Planner; use the non-diagnostic "
                "Tumor-Stroma peritumoral-focus mechanism instead."
            ),
            "melanoma-cohesive-nest-sheet": (
                "Forty top mask-ranked PUMA patches failed at the 14% visible-area floor, and retries at the "
                "5-8% compartment floor still had zero nuclei-safe interface capacity. The shared cohesive "
                "growth executor therefore cannot implement either supported primitive under the current annotations."
            ),
            "melanoma-local-tumor-clearance": (
                "The cross-validation annotation bundle has no user-supplied, digest-bound "
                "local_clearance_roi; automatic ROI invention would not be mask-authorized local clearance."
            ),
            "melanoma-intratumoral-necrosis-turnover": (
                "None of the 189 PUMA cross-validation targets has both the required Tumor-Necrosis contact "
                "and the 8% visible donor compartment for appearance or resolution. Both primitives are "
                "unavailable under the current annotation distribution."
            ),
            "melanoma-operational-tumor-retreat": (
                "Top mask-ranked PUMA retries at both the 14% visible-area floor and the 5-8% compartment "
                "floor had zero nuclei-safe Tumor-to-Stroma interface capacity. The three shared retreat "
                "programs cannot execute without cutting complete annotated nuclei."
            ),
            "melanoma-peritumoral-small-focus": (
                "Representative PUMA Tumor-Stroma boundaries could not reach the minimum four complete scatter "
                "foci, while mature small-cluster replay degraded capacity witnesses into singleton foci. Keep "
                "both primitives closed until a cluster-aware executor preserves multi-cell topology without "
                "lowering the visible-effect floor."
            ),
            "oral-scc-annotation-anchored-cord-extension": (
                "ORCA Other tissue is a heterogeneous non-carcinoma class and cannot certify a physiologically "
                "valid receiving substrate for an invasive cord without prohibited H&E interpretation."
            ),
            "oral-scc-cohesive-nest-cord": (
                "ORCA Other tissue is a heterogeneous non-carcinoma class. A mask-only Planner cannot guarantee "
                "that boundary expansion preserves mucosa, muscle, salivary tissue, nerves and vessels."
            ),
            "oral-scc-dispersed-invasive-front": (
                "ORCA Other tissue cannot identify a safe stromal invasive front. Placing carcinoma nuclei outside "
                "Tumor would risk mucosa, muscle, salivary, nerve or vascular compartments that are not encoded."
            ),
            "oral-scc-local-carcinoma-clearance": (
                "The cross-validation annotation bundle has no user-supplied, digest-bound local_clearance_roi, "
                "and ORCA has no specific receiving stroma label."
            ),
            "oral-scc-operational-tumor-retreat": (
                "ORCA has no explicit stroma or treatment-bed class. Tumor-to-Other-tissue replacement would "
                "overclaim post-treatment physiology from a heterogeneous annotation label."
            ),
        }
    )
    implementation_mechanisms = {
        "lung-generic-immune-compartment-turnover",
        "lung-intratumoral-necrosis-turnover",
        "lung-operational-tumor-retreat",
        "lung-solid-squamous-growth",
        "lung-stromal-invasive-front",
        "melanoma-cohesive-nest-sheet",
        "melanoma-operational-tumor-retreat",
        "melanoma-peritumoral-small-focus",
    }
    for mechanism_id in implementation_mechanisms:
        closed_mechanisms.pop(mechanism_id, None)

    target_closure_categories = {
        "lung-acinar-papillary-growth": "annotation_limited",
        "lung-lepidic-growth": "annotation_limited",
        "lung-local-tumor-clearance": "annotation_limited",
        "lung-stas-airspace-spread": "annotation_limited",
        "melanoma-discohesive-junctional": "annotation_limited",
        "melanoma-intratumoral-necrosis-turnover": "dataset_case_limited",
        "melanoma-local-tumor-clearance": "annotation_limited",
        "oral-scc-annotation-anchored-cord-extension": "annotation_limited",
        "oral-scc-cohesive-nest-cord": "annotation_limited",
        "oral-scc-dispersed-invasive-front": "annotation_limited",
        "oral-scc-local-carcinoma-clearance": "annotation_limited",
        "oral-scc-operational-tumor-retreat": "annotation_limited",
    }
    categories = execution_scope.setdefault("closed_mechanism_categories", {})
    target_prefixes = ("lung-", "melanoma-", "oral-scc-")
    for mechanism_id in tuple(categories):
        if mechanism_id.startswith(target_prefixes):
            categories.pop(mechanism_id, None)
    categories.update(target_closure_categories)
    execution_scope["closed_pair_categories"] = {
        pair_id: category
        for pair_id, category in execution_scope.get(
            "closed_pair_categories", {}
        ).items()
        if pair_id in closed_pairs
    }
    writer.json(execution_scope_path, execution_scope)

    # P1: complete PANDA cell-only dispersion and post-treatment residual scope.
    prostate_local = _contract(root, "prostate-local-population-modulation")
    scatter_primitive = "peritumoral-neoplastic-scatter-increase-v1"
    prostate_local["supported_primitives"] = [
        item
        for item in prostate_local["supported_primitives"]
        if item != scatter_primitive
    ]
    prostate_local["tissue_program"]["primitive_label_contracts"].pop(
        scatter_primitive, None
    )
    prostate_local["cell_program"]["layout_program_by_primitive"].pop(
        scatter_primitive, None
    )
    prostate_local["coupling_contract"][
        "allow_neoplastic_in_non_tumor_tissue"
    ] = False
    _write_contract(writer, prostate_local)
    _new_mechanism(
        writer,
        template_id="breast-peritumoral-neoplastic-scatter",
        mechanism_id="prostate-pattern-5-peripheral-scatter",
        domain_id="prostate-adenocarcinoma-v1",
        transform=_prostate_pattern5_scatter_transform,
        pathology_sources=["pathology-prostate-pattern5-2015"],
        skill_text=(
            "---\n"
            "name: prostate-pattern-5-peripheral-scatter\n"
            "description: Shadow-only PANDA fine-10 peripheral class-1 scatter with native lumen protection.\n"
            "---\n\n"
            "# PANDA Pattern-5 peripheral scatter\n\n"
            "Require an explicit fine-10/fine-2 boundary anchor and the "
            "deterministically produced native pattern/lumen protection map. "
            "Add complete sparse class-1 instances only; do not claim a grade "
            "change, histologic invasion, extraprostatic extension or prognosis.\n"
        ),
    )
    prostate_retreat = _contract(root, "prostate-operational-tumor-retreat")
    _retreat_transform(prostate_retreat)
    _write_contract(writer, prostate_retreat)
    _write_shadow_skill(
        writer,
        "prostate-operational-tumor-retreat",
        title="PANDA operational tumor-to-stroma turnover",
        body=(
            "Require explicit post-treatment semantic intent and fine-9/10 source "
            "authority. Fine-2 is only operational Stroma: it does not prove "
            "fibrosis, tumor bed, pCR, response, prognosis or clinical benefit."
        ),
        display_name="PANDA operational tumor-to-stroma turnover",
    )

    # P2 IGNITE.
    lung_growth = _contract(root, "lung-solid-squamous-growth")
    _generic_boundary_growth(lung_growth, host="Stroma")
    _remove_redundant_tumor_burden_growth(lung_growth)
    _write_contract(writer, lung_growth)
    _write_shadow_skill(
        writer,
        "lung-solid-squamous-growth",
        title="IGNITE annotation-anchored boundary growth",
        body=(
            "Select only certified external fine-1/fine-2 mask boundaries. The "
            "coarse IGNITE Tumor label does not authorize squamous, solid, acinar, "
            "papillary or lepidic identity, and raw H&E is not an execution source."
        ),
    )
    lung_cord = _contract(root, "lung-stromal-invasive-front")
    _cord(lung_cord, host="Stroma")
    _write_contract(writer, lung_cord)
    _write_shadow_skill(
        writer,
        "lung-stromal-invasive-front",
        title="IGNITE annotation-anchored synthetic cord extension",
        body=(
            "Use one compiler-certified fine-1/fine-2 external boundary and the "
            "directional tapered executor. This is narrow connected mask geometry, "
            "not a histologic invasive-front diagnosis or subtype claim."
        ),
    )
    lung_local = _contract(root, "lung-local-population-modulation")
    _add_local_cell_primitives(lung_local, include_generic_inflammatory=True)
    _cell_only_dispersion(lung_local, host_label="Stroma", include_cluster=True)
    _remove_mixed_scope_dispersion_checks(lung_local)
    # Lung inflammatory nuclei often occupy a sparse alveolar-interstitial
    # field.  A bounded multi-focus whole-instance decrement remains
    # meaningful without inventing a dense radial core/transition gradient.
    lung_local["cell_program"]["layout_program_by_primitive"][
        "generic-inflammatory-cell-abundance-decrease-v1"
    ] = "single"
    lung_local["cell_program"]["cellularity_depletion_contract"][
        "minimum_field_area_cell_diameter_squares"
    ] = 6
    lung_local["cell_program"]["cellularity_depletion_contract"][
        "outer_reference_width_cell_diameters"
    ] = 3
    lung_local["cell_program"]["cellularity_depletion_contract"][
        "minimum_outer_reference_instances"
    ] = 0
    lung_local["cell_program"]["cellularity_depletion_contract"][
        "allowed_neighbor_labels"
    ] = _ordered_union(
        lung_local["cell_program"]["cellularity_depletion_contract"][
            "allowed_neighbor_labels"
        ],
        ["Tumor"],
    )
    lung_local["cell_program"]["cellularity_depletion_contract"][
        "allowed_anchor_types"
    ] = _ordered_union(
        lung_local["cell_program"]["cellularity_depletion_contract"][
            "allowed_anchor_types"
        ],
        ["population_peak"],
    )
    lung_local["cell_program"]["cellularity_depletion_contract"][
        "core_width_cell_diameters"
    ] = 2.5
    _write_contract(writer, lung_local)
    lung_retreat = _contract(root, "lung-operational-tumor-retreat")
    _retreat_transform(lung_retreat)
    _write_contract(writer, lung_retreat)
    _write_shadow_skill(
        writer,
        "lung-operational-tumor-retreat",
        title="IGNITE post-treatment operational tumor retreat",
        body=(
            "Require explicit post-treatment context and certified fine-1 to "
            "fine-2 candidates. Fine-2 does not prove fibrosis, tumor bed, major "
            "pathologic response, pCR, prognosis or benefit."
        ),
    )

    # P2 PUMA.
    melanoma_growth = _contract(root, "melanoma-cohesive-nest-sheet")
    _generic_boundary_growth(melanoma_growth, host="Stroma")
    _remove_redundant_tumor_burden_growth(melanoma_growth)
    _write_contract(writer, melanoma_growth)
    _write_shadow_skill(
        writer,
        "melanoma-cohesive-nest-sheet",
        title="PUMA annotation-anchored melanoma boundary growth",
        body=(
            "Use only an explicit Tumor/Stroma external mask boundary and protect "
            "epidermis, vessel, necrosis and zero. Do not infer nest/sheet identity, "
            "melanocytic cytology, microsatellite status or prognosis from H&E."
        ),
    )
    melanoma_scatter = _contract(root, "melanoma-discohesive-junctional")
    melanoma_scatter["supported_primitives"] = []
    melanoma_scatter["tissue_program"]["primitive_label_contracts"] = {}
    melanoma_scatter["cell_program"]["layout_program_by_primitive"] = {}
    _cell_only_dispersion(melanoma_scatter, host_label="Stroma", include_cluster=False)
    melanoma_scatter["cell_program"]["required_checker_ids"] = _ordered_union(
        melanoma_scatter["cell_program"]["required_checker_ids"],
        ["puma_epidermal_junction_binding"],
    )
    melanoma_scatter["coupling_contract"].update(
        joint_area_mode="cell_count_extent",
        tissue_floor_applies=False,
        cell_only_target_fraction=1,
    )
    _write_contract(writer, melanoma_scatter)
    _write_shadow_skill(
        writer,
        "melanoma-discohesive-junctional",
        title="PUMA synthetic epidermal-junction neoplastic scatter",
        body=(
            "Require the explicit Epidermis label and digest-bound junction map. "
            "Add complete class-1 singles only in the certified band; do not "
            "diagnose pagetoid spread, radial growth or melanocytic identity."
        ),
    )
    _new_mechanism(
        writer,
        template_id="melanoma-discohesive-junctional",
        mechanism_id="melanoma-peritumoral-small-focus",
        domain_id="melanoma-v1",
        transform=_melanoma_small_focus_transform,
        pathology_sources=["pathology-melanoma-cap-v2025"],
        skill_text=(
            "---\n"
            "name: melanoma-peritumoral-small-focus\n"
            "description: Shadow-only PUMA stromal-annulus small neoplastic foci.\n"
            "---\n\n"
            "# PUMA peritumoral small focus\n\n"
            "Add multiple separated complete 1-4-cell class-1 foci only in a "
            "certified Tumor/Stroma outer annulus. Preserve epidermis, vessel, "
            "necrosis and zero. Never call the output a microsatellite, "
            "metastasis, staging feature or prognostic finding.\n"
        ),
    )
    melanoma_local = _contract(root, "melanoma-local-population-modulation")
    _add_local_cell_primitives(melanoma_local, include_generic_inflammatory=True)
    melanoma_local["cell_program"]["cellularity_depletion_contract"][
        "minimum_field_area_cell_diameter_squares"
    ] = 48
    melanoma_local["cell_program"]["cellularity_depletion_contract"][
        "outer_reference_width_cell_diameters"
    ] = 3
    melanoma_local["cell_program"]["cellularity_depletion_contract"][
        "minimum_outer_reference_instances"
    ] = 0
    melanoma_local["cell_program"]["cellularity_depletion_contract"][
        "allowed_neighbor_labels"
    ] = _ordered_union(
        melanoma_local["cell_program"]["cellularity_depletion_contract"][
            "allowed_neighbor_labels"
        ],
        ["Tumor"],
    )
    melanoma_local["cell_program"]["cellularity_depletion_contract"].update(
        # PUMA class-2 depletion must be a visible local density change, not a
        # handful of nearly imperceptible nuclei.  The retained residual and
        # outer reference still prohibit near-total inflammatory clearance.
        core_width_cell_diameters=4,
        transition_width_cell_diameters=8,
        transition_subband_count=6,
        core_target_removal_fraction=0.65,
        transition_start_removal_fraction=0.50,
        transition_end_removal_fraction=0.10,
        minimum_core_residual_fraction=0.32,
        minimum_transition_residual_fraction=0.45,
    )
    melanoma_local["cell_program"]["cellularity_depletion_contract"][
        "allowed_anchor_types"
    ] = _ordered_union(
        melanoma_local["cell_program"]["cellularity_depletion_contract"][
            "allowed_anchor_types"
        ],
        ["population_peak"],
    )
    _write_contract(writer, melanoma_local)

    # P2 ORCA.
    oral_growth = _contract(root, "oral-scc-cohesive-nest-cord")
    _generic_boundary_growth(oral_growth, host="Other tissue")
    oral_growth["supported_primitives"] = [
        "tumor-burden-increase-v1",
        "cohesive-boundary-expansion-v1",
    ]
    oral_growth["tissue_program"]["primitive_label_contracts"] = {
        primitive_id: {
            "source_labels": ["Other tissue"],
            "target_labels": ["Tumor"],
        }
        for primitive_id in oral_growth["supported_primitives"]
    }
    oral_growth["cell_program"]["layout_program_by_primitive"] = {
        primitive_id: "boundary_aligned"
        for primitive_id in oral_growth["supported_primitives"]
    }
    oral_growth["tissue_program"]["allowed_tools"] = [
        item
        for item in oral_growth["tissue_program"]["allowed_tools"]
        if item != "directional_tapered_projection"
    ]
    oral_growth["tissue_program"]["front_contract"] = copy.deepcopy(
        _contract(root, "breast-annotation-anchored-boundary-growth")[
            "tissue_program"
        ]["front_contract"]
    )
    _write_contract(writer, _sanitize_orca_language(oral_growth))
    _write_shadow_skill(
        writer,
        "oral-scc-cohesive-nest-cord",
        title="ORCA annotation-anchored carcinoma boundary growth",
        body=(
            "Use only certified Tumor/Other-tissue external boundaries and preserve "
            "every zero pixel. Other tissue is heterogeneous, not stroma. Do not "
            "claim invasive subtype, keratinized nest, WPOI, PNI, LVI or prognosis."
        ),
    )
    oral_scatter = _contract(root, "oral-scc-dispersed-invasive-front")
    oral_scatter["supported_primitives"] = []
    oral_scatter["tissue_program"]["primitive_label_contracts"] = {}
    oral_scatter["cell_program"]["layout_program_by_primitive"] = {}
    _cell_only_dispersion(oral_scatter, host_label="Other tissue", include_cluster=True)
    _oral_scatter_transform(oral_scatter)
    oral_scatter["coupling_contract"].update(
        joint_area_mode="cell_count_extent",
        tissue_floor_applies=False,
        cell_only_target_fraction=1,
    )
    _write_contract(writer, _sanitize_orca_language(oral_scatter))
    _write_shadow_skill(
        writer,
        "oral-scc-dispersed-invasive-front",
        title="ORCA synthetic pericarcinoma scatter and small clusters",
        body=(
            "Add complete class-1 singles or 1–4-cell foci in the certified "
            "Tumor/Other-tissue outer annulus while preserving ORCA zero exactly. "
            "This is not diagnostic budding, WPOI, stromal invasion or prognosis."
        ),
    )
    oral_local = _contract(root, "oral-scc-local-population-modulation")
    _add_local_cell_primitives(
        oral_local,
        include_generic_inflammatory=True,
        generic_host_labels=("Tumor", "Other tissue"),
    )
    oral_local["cell_program"]["cellularity_depletion_contract"][
        "minimum_field_area_cell_diameter_squares"
    ] = 48
    oral_local["cell_program"]["cellularity_depletion_contract"][
        "outer_reference_width_cell_diameters"
    ] = 3
    oral_local["cell_program"]["cellularity_depletion_contract"][
        "minimum_outer_reference_instances"
    ] = 0
    oral_local["cell_program"]["cellularity_depletion_contract"][
        "allowed_neighbor_labels"
    ] = _ordered_union(
        oral_local["cell_program"]["cellularity_depletion_contract"][
            "allowed_neighbor_labels"
        ],
        ["Tumor"],
    )
    oral_local["cell_program"]["cellularity_depletion_contract"].update(
        # A wider, stronger center-to-periphery gradient represents an
        # observable immune-cell-poor field while preserving both a residual
        # population and an unedited outer reference.  ORCA cannot support a
        # subtype, immune-exclusion, response, or prognostic interpretation.
        core_width_cell_diameters=4,
        transition_width_cell_diameters=8,
        # Six subbands retain a smooth readable gradient without making sparse
        # ORCA fields fail on fractional whole-instance quotas.
        transition_subband_count=6,
        core_target_removal_fraction=0.65,
        transition_start_removal_fraction=0.50,
        transition_end_removal_fraction=0.10,
        minimum_core_residual_fraction=0.32,
        minimum_transition_residual_fraction=0.45,
    )
    oral_local["cell_program"]["cellularity_depletion_contract"][
        "allowed_anchor_types"
    ] = _ordered_union(
        oral_local["cell_program"]["cellularity_depletion_contract"][
            "allowed_anchor_types"
        ],
        ["population_peak"],
    )
    _write_contract(writer, _sanitize_orca_language(oral_local))
    oral_local_evidence_path = (
        MECHANISMS
        / "oral-scc-local-population-modulation"
        / "references"
        / "evidence.json"
    )
    oral_local_evidence = _load(writer.root / oral_local_evidence_path)
    oral_local_evidence["records"][0]["source_ids"] = _ordered_union(
        oral_local_evidence["records"][0]["source_ids"],
        ["pathology-oral-immune-spatial-2021"],
    )
    writer.json(oral_local_evidence_path, oral_local_evidence)

    _new_mechanism(
        writer,
        template_id="breast-infiltrative-nest-cord-extension",
        mechanism_id="oral-scc-annotation-anchored-cord-extension",
        domain_id="oral-squamous-cell-carcinoma-v1",
        transform=_oral_cord_transform,
        pathology_sources=["pathology-oral-cap-v2024"],
        skill_text=(
            "---\n"
            "name: oral-scc-annotation-anchored-cord-extension\n"
            "description: Shadow-only ORCA annotation-anchored narrow connected carcinoma extension.\n"
            "---\n\n"
            "# Oral-SCC annotation-anchored cord extension\n\n"
            "Use only certified Tumor/Other-tissue external boundaries and keep "
            "every ORCA zero pixel exact. The output is synthetic cord geometry, "
            "not diagnostic invasion, WPOI, budding grade or prognosis.\n"
        ),
    )

    retreat_specs = (
        (
            "melanoma-operational-tumor-retreat",
            "melanoma-v1",
            "pathology-melanoma-cap-v2025",
        ),
        (
            "oral-scc-operational-tumor-retreat",
            "oral-squamous-cell-carcinoma-v1",
            "pathology-oral-cap-v2024",
        ),
    )
    for mechanism_id, domain_id, source_id in retreat_specs:
        _new_mechanism(
            writer,
            template_id="breast-post-treatment-invasive-regression",
            mechanism_id=mechanism_id,
            domain_id=domain_id,
            transform=_retreat_transform,
            pathology_sources=[source_id],
            skill_text=(
                "---\n"
                f"name: {mechanism_id}\n"
                "description: Shadow-only annotation-defined tumor retreat into an operational non-tumor label.\n"
                "---\n\n"
                f"# {mechanism_id}\n\n"
                "Require explicit post-treatment semantic intent and certified "
                "mask candidates. The result is operational label turnover only; "
                "do not claim fibrosis, tumor bed, pCR, response percentage, "
                "metastasis, microsatellite status or clinical benefit.\n"
            ),
        )

    clearance_specs = (
        (
            "prostate-local-tumor-clearance",
            "prostate-adenocarcinoma-v1",
            "pathology-prostate-cap-treatment-v2023",
        ),
        ("lung-local-tumor-clearance", "lung-carcinoma-v1", "pathology-lung-cap-v2025"),
        ("melanoma-local-tumor-clearance", "melanoma-v1", "pathology-melanoma-cap-v2025"),
        (
            "oral-scc-local-carcinoma-clearance",
            "oral-squamous-cell-carcinoma-v1",
            "pathology-oral-cap-v2024",
        ),
    )
    for mechanism_id, domain_id, source_id in clearance_specs:
        _new_mechanism(
            writer,
            template_id="breast-local-invasive-clearance",
            mechanism_id=mechanism_id,
            domain_id=domain_id,
            transform=_clearance_transform,
            pathology_sources=[source_id],
            skill_text=(
                "---\n"
                f"name: {mechanism_id}\n"
                "description: Shadow-only ROI-bound annotation tumor clearance.\n"
                "---\n\n"
                f"# {mechanism_id}\n\n"
                "Require a digest-bound `local_clearance_roi`. Every changed "
                "pixel and changed-instance center must remain inside it. This "
                "does not claim a negative margin, pCR or complete response.\n"
            ),
        )

    _new_mechanism(
        writer,
        template_id="breast-generic-immune-compartment-turnover",
        mechanism_id="lung-generic-immune-compartment-turnover",
        domain_id="lung-carcinoma-v1",
        transform=_immune_transform,
        pathology_sources=["pathology-lung-cap-v2025"],
        skill_text=(
            "---\n"
            "name: lung-generic-immune-compartment-turnover\n"
            "description: Shadow-only turnover of an existing IGNITE Stroma/Immune interface.\n"
            "---\n\n"
            "# Lung generic immune-compartment turnover\n\n"
            "Require an existing fine-2/fine-4 interface and complete class-2/3 "
            "references. Do not infer immune subtype, PD-L1, TIL score, response, "
            "prognosis or benefit.\n"
        ),
    )

    _update_profile(
        writer,
        profile_id="panda-gleason-v1",
        conditional=[
            "prostate-local-tumor-clearance",
            "prostate-pattern-5-peripheral-scatter",
        ],
        sources={"prostate-local-tumor-clearance": [9, 10]},
        targets={"prostate-local-tumor-clearance": [2]},
        operational_stroma_ids=[2],
        required={"prostate-pattern-5-peripheral-scatter": [10]},
    )
    _update_profile(
        writer,
        profile_id="ignite-semantic-v1",
        conditional=[
            "lung-generic-immune-compartment-turnover",
            "lung-local-tumor-clearance",
        ],
        sources={
            "lung-solid-squamous-growth": [2],
            "lung-stromal-invasive-front": [2],
            "lung-operational-tumor-retreat": [1],
            "lung-generic-immune-compartment-turnover": [2, 4],
            "lung-generic-immune-compartment-turnover::generic-immune-infiltrate-increase-v1": [2],
            "lung-generic-immune-compartment-turnover::generic-immune-infiltrate-decrease-v1": [4],
            "lung-local-tumor-clearance": [1],
        },
        targets={
            "lung-solid-squamous-growth": [1],
            "lung-stromal-invasive-front": [1],
            "lung-operational-tumor-retreat": [2],
            "lung-generic-immune-compartment-turnover": [2, 4],
            "lung-generic-immune-compartment-turnover::generic-immune-infiltrate-increase-v1": [4],
            "lung-generic-immune-compartment-turnover::generic-immune-infiltrate-decrease-v1": [2],
            "lung-local-tumor-clearance": [2],
        },
        operational_stroma_ids=[2],
    )
    _update_profile(
        writer,
        profile_id="puma-semantic-v1",
        conditional=[
            "melanoma-operational-tumor-retreat",
            "melanoma-local-tumor-clearance",
            "melanoma-peritumoral-small-focus",
        ],
        sources={
            "melanoma-cohesive-nest-sheet": [2],
            "melanoma-operational-tumor-retreat": [1],
            "melanoma-local-tumor-clearance": [1],
        },
        targets={
            "melanoma-cohesive-nest-sheet": [1],
            "melanoma-operational-tumor-retreat": [2],
            "melanoma-local-tumor-clearance": [2],
        },
        operational_stroma_ids=[2],
    )
    _update_profile(
        writer,
        profile_id="orca-semantic-v1",
        conditional=[
            "oral-scc-annotation-anchored-cord-extension",
            "oral-scc-operational-tumor-retreat",
            "oral-scc-local-carcinoma-clearance",
        ],
        sources={
            "oral-scc-cohesive-nest-cord": [7],
            "oral-scc-annotation-anchored-cord-extension": [7],
            "oral-scc-operational-tumor-retreat": [1],
            "oral-scc-local-carcinoma-clearance": [1],
        },
        targets={
            "oral-scc-cohesive-nest-cord": [1],
            "oral-scc-annotation-anchored-cord-extension": [1],
            "oral-scc-operational-tumor-retreat": [7],
            "oral-scc-local-carcinoma-clearance": [7],
        },
        operational_stroma_ids=None,
    )

    governance_path = CATALOG / "evidence-governance-v2.json"
    governance = _load(root / governance_path)
    governance["sources"]["pathology-cellvit-five-class-taxonomy-2024"] = {
        "category": "pathology_fact",
        "title": "CellViT: Vision Transformers for precise cell segmentation and classification",
        "uri": "https://pubmed.ncbi.nlm.nih.gov/38507894/",
        "locator": (
            "Pan-cancer nucleus-instance taxonomy supporting neoplastic, inflammatory, "
            "connective, dead and epithelial observation classes; these classes authorize "
            "only generic local population edits, not disease subtype or prognosis claims"
        ),
        "verification_status": "verified",
    }
    governance["sources"]["pathology-oral-immune-spatial-2021"] = {
        "category": "pathology_fact",
        "title": "B-cell clusters at the invasive margin associate with longer survival in early-stage oral-tongue cancer patients",
        "uri": "https://pubmed.ncbi.nlm.nih.gov/33643695/",
        "locator": (
            "Primary oral-tongue SCC study measuring distinct lymphocyte densities "
            "and spatial distributions in tumor and stroma at the invasive margin "
            "and tumor center. The editor uses only the existence of localized "
            "density variation and makes no immune-subtype or prognostic claim."
        ),
        "verification_status": "verified",
    }
    governance["mechanism_pathology_sources"].update(
        {
            "lung-local-population-modulation": [
                "pathology-cellvit-five-class-taxonomy-2024",
                "pathology-lung-cap-v2025",
            ],
            "lung-generic-immune-compartment-turnover": ["pathology-lung-cap-v2025"],
            "lung-local-tumor-clearance": ["pathology-lung-cap-v2025"],
            "melanoma-intratumoral-necrosis-turnover": [
                "pathology-melanoma-cap-v2025"
            ],
            "melanoma-local-population-modulation": [
                "pathology-cellvit-five-class-taxonomy-2024",
                "pathology-melanoma-cap-v2025",
            ],
            "melanoma-operational-tumor-retreat": ["pathology-melanoma-cap-v2025"],
            "melanoma-local-tumor-clearance": ["pathology-melanoma-cap-v2025"],
            "melanoma-peritumoral-small-focus": ["pathology-melanoma-cap-v2025"],
            "oral-scc-operational-tumor-retreat": ["pathology-oral-cap-v2024"],
            "oral-scc-local-population-modulation": [
                "pathology-cellvit-five-class-taxonomy-2024",
                "pathology-oral-cap-v2024",
                "pathology-oral-immune-spatial-2021",
            ],
            "oral-scc-annotation-anchored-cord-extension": ["pathology-oral-cap-v2024"],
            "oral-scc-local-carcinoma-clearance": ["pathology-oral-cap-v2024"],
            "prostate-local-tumor-clearance": ["pathology-prostate-cap-treatment-v2023"],
            "prostate-pattern-5-peripheral-scatter": ["pathology-prostate-pattern5-2015"],
        }
    )
    writer.json(governance_path, governance)
    if not check or not writer.changed:
        _assert_clean_non_breast_catalog(root)
    return writer.changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    changed = refine(root, check=args.check)
    if args.check and changed:
        for path in changed:
            print(path.relative_to(root))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

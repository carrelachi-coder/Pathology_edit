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
from pathlib import Path
from typing import Any, Callable


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
    writer: Writer, mechanism_id: str, *, title: str, body: str
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
        f'  display_name: "{mechanism_id}"\n'
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


def _add_local_cell_primitives(
    contract: dict[str, Any], *, include_generic_inflammatory: bool
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
                "source_labels": ["Tumor", "Stroma", "Other tissue"],
                "target_labels": ["Tumor", "Stroma", "Other tissue"],
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


def _cord(contract: dict[str, Any], *, host: str) -> None:
    breast = contract.pop("_breast_template")
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
    contract["tissue_program"] = copy.deepcopy(breast["tissue_program"])
    contract["tissue_program"]["mode"] = "annotation_anchored_narrow_connected_extension"
    contract["tissue_program"]["primitive_label_contracts"] = {
        "infiltrative-nest-cord-extension-v1": {
            "source_labels": [host],
            "target_labels": ["Tumor"],
        }
    }
    contract["tissue_program"]["required_checker_ids"] = [
        "tissue_gate_binding",
        "external_boundary_binding",
        "annotation_anchored_extension_geometry",
        "profile_fine_transition_authority",
    ]
    contract["cell_program"] = copy.deepcopy(breast["cell_program"])
    contract["coupling_contract"] = copy.deepcopy(breast["coupling_contract"])
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
    """Separate stromal small foci from epidermis-bound junctional singles."""

    primitive_id = "peritumoral-small-cluster-increase-v1"
    contract["supported_primitives"] = [primitive_id]
    contract["summary"] = (
        "Add multiple separated complete class-1 small foci in a certified "
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
    }
    contract["cell_program"]["layout_program_by_primitive"] = {
        primitive_id: "small_cluster"
    }
    contract["cell_program"]["layout_programs"] = ["small_cluster"]
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
            "multiple separated 1-4-cell synthetic foci in the bounded stromal annulus",
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


def _update_profile(
    writer: Writer,
    *,
    profile_id: str,
    conditional: list[str],
    sources: dict[str, list[int]],
    targets: dict[str, list[int]],
    operational_stroma_ids: list[int] | None,
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


def refine(root: Path, *, check: bool) -> list[Path]:
    writer = Writer(root, check=check)
    _install_primitives(writer)
    execution_scope_path = CATALOG / "execution-scope-v1.json"
    execution_scope = _load(root / execution_scope_path)
    execution_scope["executable_primitives"] = _ordered_union(
        execution_scope["executable_primitives"],
        [
            "generic-inflammatory-cell-abundance-increase-v1",
            "generic-inflammatory-cell-abundance-decrease-v1",
        ],
    )
    writer.json(execution_scope_path, execution_scope)

    # P1: complete PANDA cell-only dispersion and post-treatment residual scope.
    prostate_local = _contract(root, "prostate-local-population-modulation")
    _cell_only_dispersion(prostate_local, host_label="Stroma", include_cluster=False)
    _remove_mixed_scope_dispersion_checks(prostate_local)
    _write_contract(writer, prostate_local)
    prostate_retreat = _contract(root, "prostate-treatment-associated-fibrotic-replacement")
    _retreat_transform(prostate_retreat)
    _write_contract(writer, prostate_retreat)
    _write_shadow_skill(
        writer,
        "prostate-treatment-associated-fibrotic-replacement",
        title="PANDA operational tumor-to-stroma turnover",
        body=(
            "Require explicit post-treatment semantic intent and fine-9/10 source "
            "authority. Fine-2 is only operational Stroma: it does not prove "
            "fibrosis, tumor bed, pCR, response, prognosis or clinical benefit."
        ),
    )

    # P2 IGNITE.
    lung_growth = _contract(root, "lung-solid-squamous-growth")
    _generic_boundary_growth(lung_growth, host="Stroma")
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
    lung_cord["_breast_template"] = _contract(
        root, "breast-infiltrative-nest-cord-extension"
    )
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
    _write_contract(writer, lung_local)
    lung_retreat = _contract(root, "lung-treatment-associated-fibrotic-replacement")
    _retreat_transform(lung_retreat)
    _write_contract(writer, lung_retreat)
    _write_shadow_skill(
        writer,
        "lung-treatment-associated-fibrotic-replacement",
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
    _write_contract(writer, melanoma_local)

    # P2 ORCA.
    oral_growth = _contract(root, "oral-scc-cohesive-nest-cord")
    _generic_boundary_growth(oral_growth, host="Other tissue")
    oral_growth["supported_primitives"] = [
        item
        for item in oral_growth["supported_primitives"]
        if item != "infiltrative-nest-cord-extension-v1"
    ]
    oral_growth["tissue_program"]["primitive_label_contracts"].pop(
        "infiltrative-nest-cord-extension-v1", None
    )
    oral_growth["cell_program"]["layout_program_by_primitive"].pop(
        "infiltrative-nest-cord-extension-v1", None
    )
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
    _write_contract(writer, oral_growth)
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
    oral_scatter["coupling_contract"].update(
        joint_area_mode="cell_count_extent",
        tissue_floor_applies=False,
        cell_only_target_fraction=1,
    )
    _write_contract(writer, oral_scatter)
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
    _add_local_cell_primitives(oral_local, include_generic_inflammatory=True)
    _write_contract(writer, oral_local)

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
        conditional=["prostate-local-tumor-clearance"],
        sources={"prostate-local-tumor-clearance": [9, 10]},
        targets={"prostate-local-tumor-clearance": [2]},
        operational_stroma_ids=[2],
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
            "lung-treatment-associated-fibrotic-replacement": [1],
            "lung-generic-immune-compartment-turnover": [2, 4],
            "lung-generic-immune-compartment-turnover::generic-immune-infiltrate-increase-v1": [2],
            "lung-generic-immune-compartment-turnover::generic-immune-infiltrate-decrease-v1": [4],
            "lung-local-tumor-clearance": [1],
        },
        targets={
            "lung-solid-squamous-growth": [1],
            "lung-stromal-invasive-front": [1],
            "lung-treatment-associated-fibrotic-replacement": [2],
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
    governance["mechanism_pathology_sources"].update(
        {
            "lung-generic-immune-compartment-turnover": ["pathology-lung-cap-v2025"],
            "lung-local-tumor-clearance": ["pathology-lung-cap-v2025"],
            "melanoma-operational-tumor-retreat": ["pathology-melanoma-cap-v2025"],
            "melanoma-local-tumor-clearance": ["pathology-melanoma-cap-v2025"],
            "melanoma-peritumoral-small-focus": ["pathology-melanoma-cap-v2025"],
            "oral-scc-operational-tumor-retreat": ["pathology-oral-cap-v2024"],
            "oral-scc-annotation-anchored-cord-extension": ["pathology-oral-cap-v2024"],
            "oral-scc-local-carcinoma-clearance": ["pathology-oral-cap-v2024"],
            "prostate-local-tumor-clearance": ["pathology-prostate-cap-treatment-v2023"],
        }
    )
    writer.json(governance_path, governance)
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

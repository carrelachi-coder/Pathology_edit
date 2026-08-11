"""Auditable H&E qualification decisions for the frozen legacy G2 pairs.

This module does not generate masks.  It converts a completed source-only
contact-sheet review into one decision per case.  Broad deterministic policy
is used only where the H&E review confirmed a single compatible mechanism for
the whole stratum; every exception is listed explicitly below.  The output
retains the source board/page/position so a reviewer can reproduce the visual
decision instead of trusting an organ-level default.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .g2_pilot import ORGAN_CONTRACTS
from .semantic_parser import (
    SEMANTIC_INTENT_SCHEMA_VERSION,
    semantic_intent_from_metadata,
)

HE_REVIEW_SCHEMA_VERSION = "g2-he-mechanism-review-v3"
REVIEW_POLICY_VERSION = "current-codex-source-he-review-2026-08-11-v3"

DECISION_STATUSES = frozenset(
    {
        "supported_mechanism",
        "rewrite_instruction",
        "convert_cell_only",
        "replace_primitive",
        "abstain",
    }
)

LOCAL_POPULATION_MECHANISM = {
    "breast": "breast-local-population-modulation",
    "colorectal": "colorectal-local-population-modulation",
    "lung": "lung-local-population-modulation",
    "oral": "oral-scc-local-population-modulation",
    "prostate": "prostate-local-population-modulation",
    "skin": "melanoma-local-population-modulation",
}

DEFAULT_TISSUE_MECHANISM = {
    "breast": "breast-cohesive-nst-front",
    "colorectal": "colorectal-gland-forming-front",
    "oral": "oral-scc-cohesive-nest-cord",
    "skin": "melanoma-cohesive-nest-sheet",
}

NECROSIS_MECHANISM = {
    "breast": "breast-intratumoral-necrosis-turnover",
    "lung": "lung-intratumoral-necrosis-turnover",
    "skin": "melanoma-intratumoral-necrosis-turnover",
}

# H&E-reviewed lung architecture.  These are not organ defaults: the IDs bind
# the visual decision to the source board reviewed in the current Codex
# session.  Cases absent from a set fail closed.
LUNG_ACINAR_CASES = frozenset({203, 204, 205, 207, 208, 239})
LUNG_SOLID_CASES = frozenset(
    {201, 202, 206, 209, 210, *range(211, 221), 221, 226,
     231, 232, 234, 235, 236, 238, 240}
)
LUNG_STROMA_INCREASE_ABSTAIN = frozenset({222, 223, 224, 225, 227, 228, 229, 230})
LUNG_STROMA_DECREASE_ABSTAIN = frozenset({233, 237})
LUNG_NECROSIS_APPEARANCE_SUPPORTED = frozenset(
    {271, 274, 277, 278, 281, 282, 283, 285}
)

# A sparse PUMA tumor compartment can make a 14% tissue-burden edit both
# biologically misleading and geometrically unstable.  H&E review selected a
# dispersed cell-front interpretation for these cases.
SKIN_DIRECT_INFILTRATION = frozenset({513})
SKIN_STROMA_TO_INFILTRATION = frozenset({571, 579, 585, 598, 599})


def write_g2_he_review(
    qualification_jsonl: str | Path,
    *,
    output_dir: str | Path,
    semantic_review_json: str | Path,
) -> dict[str, Any]:
    source = Path(qualification_jsonl)
    semantic_source = Path(semantic_review_json)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 600:
        raise ValueError(f"expected the frozen 600-case qualification, got {len(records)}")
    qualification_digest = _sha256(source)
    semantic_review_digest = _sha256(semantic_source)
    semantic_templates = _load_codex_semantic_review(semantic_source)
    decisions = [
        _review_record(
            item,
            qualification_digest=qualification_digest,
            semantic_review_path=semantic_source,
            semantic_review_digest=semantic_review_digest,
            semantic_templates=semantic_templates,
        )
        for item in records
    ]
    _validate_complete_review(records, decisions)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "he_mechanism_decisions.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in decisions
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": HE_REVIEW_SCHEMA_VERSION,
        "review_policy_version": REVIEW_POLICY_VERSION,
        "qualification_jsonl": str(source),
        "qualification_sha256": qualification_digest,
        "semantic_review_json": str(semantic_source),
        "semantic_review_sha256": semantic_review_digest,
        "decision_ledger": str(ledger),
        "decision_ledger_sha256": _sha256(ledger),
        "case_count": len(decisions),
        "by_status": dict(sorted(Counter(item["decision_status"] for item in decisions).items())),
        "by_organ_and_status": {
            f"{organ}:{status}": count
            for (organ, status), count in sorted(
                Counter((item["organ"], item["decision_status"]) for item in decisions).items()
            )
        },
        "abstain_reason_counts": dict(
            sorted(
                Counter(
                    item["reason_code"]
                    for item in decisions
                    if item["decision_status"] == "abstain"
                ).items()
            )
        ),
        "reviewer": "current_codex_session",
        "llm_api_used": False,
        "target_mask_created": False,
        "source_asset_mutated": False,
    }
    summary_path = root / "he_mechanism_review_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _review_record(
    record: dict[str, Any],
    *,
    qualification_digest: str,
    semantic_review_path: Path,
    semantic_review_digest: str,
    semantic_templates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_number = int(record["source_index"]) + 1
    organ = str(record["organ"])
    legacy = str(record["legacy_primitive"])
    status, primitive, mechanism, instruction, reason_code, observations = _decision(
        case_number=case_number,
        organ=organ,
        legacy_primitive=legacy,
        record=record,
    )
    if status not in DECISION_STATUSES:
        raise AssertionError(f"unregistered H&E decision status: {status}")
    board = record.get("review_board")
    if not isinstance(board, dict):
        raise ValueError(f"case {record['case_id']} is missing its source review board binding")
    semantic_intent = (
        _bind_codex_semantic_intent(
            case_id=str(record["case_id"]),
            instruction=str(instruction),
            primitive_id=str(primitive),
            qualification_digest=qualification_digest,
            semantic_review_digest=semantic_review_digest,
            semantic_templates=semantic_templates,
        )
        if status != "abstain"
        else None
    )
    semantic_digest = (
        hashlib.sha256(
            json.dumps(
                semantic_intent,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if semantic_intent is not None
        else None
    )
    return {
        "schema_version": HE_REVIEW_SCHEMA_VERSION,
        "review_policy_version": REVIEW_POLICY_VERSION,
        "case_id": record["case_id"],
        "source_index": record["source_index"],
        "organ": organ,
        "dataset": record["dataset"],
        "pathology_domain_id": record["pathology_domain_id"],
        "annotation_profile_id": record["annotation_profile_id"],
        "original_instruction": record["instruction"],
        "legacy_primitive": legacy,
        "decision_status": status,
        "selected_joint_primitive": primitive,
        "selected_mechanism_id": mechanism,
        "recommended_instruction": instruction,
        "prebound_semantic_intent": semantic_intent,
        "prebound_semantic_intent_sha256": semantic_digest,
        "reason_code": reason_code,
        "visual_observations": list(observations),
        "review_confidence": "high" if status != "abstain" else "high_for_rejection",
        "review_basis": {
            "reviewer": "current_codex_session",
            "modality": "source_he_plus_tissue_and_nuclei_overlays",
            "board_path": board["path"],
            "board_page": board["page"],
            "board_position": board["position"],
            "qualification_sha256": qualification_digest,
            "semantic_review_json": str(semantic_review_path),
            "semantic_review_sha256": semantic_review_digest,
            "source_image_sha256": record["source_assets"]["image_sha256"],
            "source_tissue_mask_sha256": record["source_assets"]["tissue_mask_sha256"],
            "source_nuclei_mask_sha256": record["source_assets"]["nuclei_mask_sha256"],
            "llm_api_used": False,
        },
        "execution_allowed": status != "abstain",
    }


def _decision(
    *,
    case_number: int,
    organ: str,
    legacy_primitive: str,
    record: dict[str, Any],
) -> tuple[str, str | None, str | None, str | None, str, tuple[str, ...]]:
    if legacy_primitive in {"stromal_immune_infiltration", "immune_infiltration_decrease"}:
        increase = legacy_primitive == "stromal_immune_infiltration"
        return (
            "convert_cell_only",
            "cell-type-abundance-increase-v1" if increase else "cell-type-abundance-decrease-v1",
            LOCAL_POPULATION_MECHANISM[organ],
            "increase immune cell infiltration" if increase else "decrease immune cell infiltration",
            "legacy_tissue_immune_edit_recast_as_cell_population",
            (
                "H&E and nuclei overlay expose an immune population rather than a structural compartment",
                "tissue architecture must remain fixed while complete immune instances change",
            ),
        )

    if organ == "colorectal" and legacy_primitive == "stromal_desmoplasia":
        return _abstain(
            "glas_non_gland_complement_has_no_stroma_authority",
            "GLaS marks the heterogeneous non-gland complement, not explicit desmoplastic stroma",
            "the H&E field cannot make a source-to-target stroma transition auditable from this mask",
        )

    if organ == "lung":
        return _lung_decision(case_number, legacy_primitive)

    if organ == "prostate":
        mechanism = _prostate_mechanism(record)
        if mechanism is None:
            return _abstain(
                "panda_pattern_authority_is_ambiguous",
                "no single native Gleason component can be selected without an implicit pattern conversion",
            )
        if legacy_primitive == "stroma_decrease":
            return (
                "replace_primitive",
                "tumor-burden-increase-v1",
                mechanism,
                "increase tumor burden",
                "stroma_decrease_replaced_by_pattern_preserving_tumor_growth",
                (
                    "H&E shows a tumor-stroma interface compatible with the selected native Gleason component",
                    "the edit must expand that fine ID without converting any other Gleason pattern",
                ),
            )
        return _supported_tissue(legacy_primitive, mechanism, "native Gleason architecture is visible and fine-ID-bound")

    if organ == "skin" and case_number in SKIN_DIRECT_INFILTRATION:
        return (
            "replace_primitive",
            "neoplastic-cell-infiltration-increase-v1",
            "melanoma-discohesive-junctional",
            "increase tumor infiltration",
            "sparse_melanoma_front_is_cellular_not_bulk_burden",
            (
                "H&E shows a sparse/discohesive melanoma population with insufficient bulk tumor authority",
                "preserve tissue compartments and extend a bounded neoplastic cell front",
            ),
        )

    if legacy_primitive == "stroma_decrease":
        if organ == "skin" and case_number in SKIN_STROMA_TO_INFILTRATION:
            return (
                "replace_primitive",
                "neoplastic-cell-infiltration-increase-v1",
                "melanoma-discohesive-junctional",
                "increase tumor infiltration",
                "stroma_decrease_replaced_by_dispersed_melanoma_infiltration",
                (
                    "H&E shows a sparse melanoma front rather than a broad cohesive tumor-stroma interface",
                    "a cell-only infiltration program is closer to the requested progression than bulk relabeling",
                ),
            )
        return (
            "replace_primitive",
            "tumor-burden-increase-v1",
            DEFAULT_TISSUE_MECHANISM[organ],
            "increase tumor burden",
            "stroma_decrease_replaced_by_biologically_named_tumor_growth",
            (
                "H&E shows an existing tumor-stroma interface",
                "the selected mechanism names the actual process instead of deleting an abstract stroma label",
            ),
        )

    if legacy_primitive in {"necrosis_appearance", "necrosis_resolution"}:
        mechanism = NECROSIS_MECHANISM.get(organ)
        if mechanism is None:
            return _abstain(
                "no_reviewed_necrosis_mechanism_for_domain",
                "the domain has no executable necrosis turnover mechanism",
            )
        return _supported_tissue(legacy_primitive, mechanism, "intratumoral necrosis is visible or has a viable intratumoral anchor")

    mechanism = DEFAULT_TISSUE_MECHANISM.get(organ)
    if mechanism is None:
        return _abstain("no_he_supported_mechanism", "no compatible reviewed mechanism was selected")
    return _supported_tissue(legacy_primitive, mechanism, "H&E confirms the mechanism-specific tissue architecture")


def _lung_decision(case_number: int, legacy: str):
    if legacy == "stromal_desmoplasia" and case_number in LUNG_STROMA_INCREASE_ABSTAIN:
        return _abstain(
            "lung_native_structure_not_desmoplastic_stroma",
            "H&E shows airway/alveolar or low-support native tissue rather than an auditable tumor-desmoplasia interface",
        )
    if legacy == "stroma_decrease" and case_number in LUNG_STROMA_DECREASE_ABSTAIN:
        return _abstain(
            "lung_no_supported_tumor_stroma_replacement_front",
            "H&E does not show enough compatible tumor-stroma interface for a biologically named replacement",
        )
    if legacy == "necrosis_appearance" and case_number not in LUNG_NECROSIS_APPEARANCE_SUPPORTED:
        return _abstain(
            "lung_airspace_or_lumen_is_not_necrosis_seed",
            "H&E shows an airway/alveolar/luminal space or lacks a defensible solid intratumoral necrosis anchor",
        )
    if legacy == "stroma_decrease":
        mechanism = _lung_growth_mechanism(case_number)
        if mechanism is None:
            return _abstain("lung_growth_pattern_unresolved", "H&E growth architecture is unresolved")
        return (
            "replace_primitive",
            "tumor-burden-increase-v1",
            mechanism,
            "increase tumor burden",
            "stroma_decrease_replaced_by_lung_pattern_preserving_growth",
            ("H&E shows a compatible tumor-stroma interface", "native airspace and airway structures remain prohibited"),
        )
    if legacy in {"necrosis_appearance", "necrosis_resolution"}:
        return _supported_tissue(legacy, "lung-intratumoral-necrosis-turnover", "H&E and the native necrosis label support intratumoral turnover")
    if legacy in {"stromal_immune_infiltration", "immune_infiltration_decrease"}:
        raise AssertionError("immune decisions are handled before lung dispatch")
    mechanism = _lung_growth_mechanism(case_number)
    if mechanism is None:
        return _abstain("lung_growth_pattern_unresolved", "H&E growth architecture is unresolved")
    return _supported_tissue(legacy, mechanism, "H&E supports the selected lung growth architecture")


def _lung_growth_mechanism(case_number: int) -> str | None:
    if case_number in LUNG_ACINAR_CASES:
        return "lung-acinar-papillary-growth"
    if case_number in LUNG_SOLID_CASES:
        return "lung-solid-squamous-growth"
    return None


def _prostate_mechanism(record: dict[str, Any]) -> str | None:
    counts = {
        int(key): int(value)
        for key, value in record["source_statistics"]["tissue"]["fine_id_counts"].items()
        if int(key) in {8, 9, 10} and int(value) > 0
    }
    if not counts:
        return None
    selected = max(counts, key=lambda key: (counts[key], key))
    return {
        8: "prostate-pattern-3-growth",
        9: "prostate-pattern-4-growth",
        10: "prostate-pattern-5-growth",
    }[selected]


def _supported_tissue(legacy: str, mechanism: str, observation: str):
    primitive = {
        "tumor_burden_increase": "tumor-burden-increase-v1",
        "tumor_burden_decrease": "tumor-burden-decrease-v1",
        "stromal_desmoplasia": "stroma-increase-v1",
        "necrosis_appearance": "necrosis-appearance-v1",
        "necrosis_resolution": "necrosis-resolution-v1",
    }.get(legacy)
    instruction = {
        "tumor_burden_increase": "increase tumor burden",
        "tumor_burden_decrease": "decrease tumor burden",
        "stromal_desmoplasia": "increase tumor-associated stroma",
        "necrosis_appearance": "increase tumor necrosis",
        "necrosis_resolution": "decrease tumor necrosis",
    }.get(legacy)
    if primitive is None:
        raise AssertionError(f"unsupported direct legacy primitive: {legacy}")
    return (
        "supported_mechanism",
        primitive,
        mechanism,
        instruction,
        "he_supports_requested_joint_mechanism",
        (observation, "mechanism-specific contraindications were not observed on the source review board"),
    )


def _abstain(reason_code: str, *observations: str):
    return ("abstain", None, None, None, reason_code, tuple(observations))


def _load_codex_semantic_review(path: Path) -> dict[str, dict[str, Any]]:
    """Load, but never infer, language decisions authored in this session."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "codex-session-semantic-review-v1":
        raise ValueError("unsupported current-Codex semantic review schema")
    if payload.get("reviewer") != "current_codex_session":
        raise ValueError("semantic review was not authored by the current Codex session")
    if payload.get("authoring_mode") != "interactive_codex_session_no_external_api":
        raise ValueError("semantic review has an invalid authoring mode")
    if payload.get("semantic_intent_schema_version") != SEMANTIC_INTENT_SCHEMA_VERSION:
        raise ValueError("semantic review targets an unsupported intent schema")
    if payload.get("scope") != "language_only_no_he_or_execution_authority":
        raise ValueError("semantic review improperly claims visual or execution authority")
    rows = payload.get("intents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("semantic review contains no language intents")
    by_instruction: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("semantic review intent is not an object")
        intent = dict(raw)
        intent["schema_version"] = SEMANTIC_INTENT_SCHEMA_VERSION
        validated = semantic_intent_from_metadata(intent).to_metadata()
        instruction = validated["instruction"]
        if instruction in by_instruction:
            raise ValueError(f"duplicate current-Codex instruction: {instruction}")
        by_instruction[instruction] = validated
    return by_instruction


def _bind_codex_semantic_intent(
    *,
    case_id: str,
    instruction: str,
    primitive_id: str,
    qualification_digest: str,
    semantic_review_digest: str,
    semantic_templates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind one session-authored language decision to a qualified case."""

    template = semantic_templates.get(instruction)
    if template is None:
        raise ValueError(
            "current Codex session has not parsed the reviewed instruction: "
            f"{instruction!r}"
        )
    intent = json.loads(json.dumps(template, ensure_ascii=False))
    if intent["primitive_id"] != primitive_id:
        raise ValueError(
            "current-Codex language decision conflicts with the H&E-reviewed "
            f"primitive for {case_id}"
        )
    metadata = dict(intent["parser_metadata"])
    metadata.update(
        {
            "reviewer": "current_codex_session",
            "case_id": case_id,
            "qualification_sha256": qualification_digest,
            "semantic_review_sha256": semantic_review_digest,
            "llm_api_used": False,
            "execution_runner_may_not_reparse": True,
        }
    )
    intent["parser_metadata"] = metadata
    return semantic_intent_from_metadata(intent).to_metadata()


def _validate_complete_review(source: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    source_ids = [item["case_id"] for item in source]
    decision_ids = [item["case_id"] for item in decisions]
    if decision_ids != source_ids or len(set(decision_ids)) != len(decision_ids):
        raise ValueError("H&E decision ledger is not a one-to-one ordered cover of qualification cases")
    for item in decisions:
        executable = item["decision_status"] != "abstain"
        fields_present = bool(item["selected_joint_primitive"] and item["selected_mechanism_id"])
        if executable != fields_present or executable != bool(item["recommended_instruction"]):
            raise ValueError(f"incomplete execution decision for {item['case_id']}")
        semantic = item.get("prebound_semantic_intent")
        semantic_digest = item.get("prebound_semantic_intent_sha256")
        if executable != isinstance(semantic, dict) or executable != bool(
            semantic_digest
        ):
            raise ValueError(
                f"incomplete Codex semantic binding for {item['case_id']}"
            )
        if executable:
            actual = hashlib.sha256(
                json.dumps(
                    semantic,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if actual != semantic_digest:
                raise ValueError(
                    f"Codex semantic digest drift for {item['case_id']}"
                )
            if semantic["instruction"] != item["recommended_instruction"]:
                raise ValueError(
                    f"Codex semantic instruction drift for {item['case_id']}"
                )
            if semantic["primitive_id"] != item["selected_joint_primitive"]:
                raise ValueError(
                    f"Codex semantic primitive drift for {item['case_id']}"
                )
        expected_domain, expected_annotation, _ = ORGAN_CONTRACTS[item["organ"]]
        if item["pathology_domain_id"] != expected_domain or item["annotation_profile_id"] != expected_annotation:
            raise ValueError(f"four-axis metadata drift for {item['case_id']}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

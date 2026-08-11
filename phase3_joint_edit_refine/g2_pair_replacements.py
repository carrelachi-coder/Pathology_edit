"""Apply digest-bound, H&E-reviewed replacements to a frozen G2-v2 manifest.

The base manifest is immutable.  A replacement is admitted only when its
source assets, deterministic exact-execution qualification and current-Codex
visual review are all bound by digest.  The output is a new manifest; this
module never rewrites either input.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .g2_v2_manifest import (
    G2_V2_MANIFEST_SCHEMA,
    PRIMITIVE_ONTOLOGY_VERSION,
)
from .g2_qualification import LEGACY_PRIMITIVE_MAP

PAIR_REPLACEMENT_LEDGER_SCHEMA = "g2-v2-pair-replacement-ledger-v1"
PAIR_REPLACEMENT_POLICY = "digest-bound-exact-and-current-codex-he-v1"
EXECUTION_QUALIFICATION_SCHEMA = (
    "g2-v2-read-only-execution-qualification-v1"
)


def apply_g2_v2_pair_replacements(
    base_manifest_path: str | Path,
    replacement_ledger_path: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    base_path = Path(base_manifest_path)
    ledger_path = Path(replacement_ledger_path)
    base_sha256 = _sha256(base_path)
    ledger_sha256 = _sha256(ledger_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _validate_top_level(
        base,
        ledger,
        base_sha256=base_sha256,
    )

    rows = copy.deepcopy(base["cases"])
    by_id = {str(item["case_id"]): index for index, item in enumerate(rows)}
    replacements = ledger["replacements"]
    final_ids = {str(item["case_id"]) for item in rows}
    replaced_targets: set[str] = set()
    audit_rows = []
    for request in replacements:
        target_case_id = _required_text(request, "target_case_id")
        final_case_id = _required_text(request, "final_case_id")
        if target_case_id in replaced_targets:
            raise ValueError(f"duplicate replacement target: {target_case_id}")
        if target_case_id not in by_id:
            raise ValueError(f"replacement target is absent: {target_case_id}")
        if final_case_id in final_ids and final_case_id != target_case_id:
            raise ValueError(f"replacement case ID already exists: {final_case_id}")
        target_index = by_id[target_case_id]
        target = rows[target_index]
        if target.get("execution_allowed"):
            raise ValueError(
                f"replacement may only target a fail-closed pair: {target_case_id}"
            )
        replacement, audit = _materialize_replacement(
            target,
            request,
            final_case_id=final_case_id,
            replacement_ledger_sha256=ledger_sha256,
        )
        rows[target_index] = replacement
        replaced_targets.add(target_case_id)
        final_ids.discard(target_case_id)
        final_ids.add(final_case_id)
        audit_rows.append(audit)

    if len({str(item["case_id"]) for item in rows}) != len(rows):
        raise ValueError("pair replacement produced duplicate final case IDs")
    for index, item in enumerate(rows):
        if int(item.get("source_index", -1)) != index:
            raise ValueError("pair replacement changed frozen source ordering")

    execution_count = sum(bool(item.get("execution_allowed")) for item in rows)
    decision_counts = Counter(str(item["decision_status"]) for item in rows)
    result = copy.deepcopy(base)
    result.update(
        {
            "manifest_id": f"g2-v2-pair-replaced-{ledger_sha256[:12]}",
            "case_count": len(rows),
            "execution_case_count": execution_count,
            "abstain_case_count": len(rows) - execution_count,
            "decision_counts": dict(sorted(decision_counts.items())),
            "cases": rows,
            "pair_replacement_policy": PAIR_REPLACEMENT_POLICY,
            "pair_replacement_count": len(audit_rows),
            "pair_replacement_audit": audit_rows,
        }
    )
    source_chain = dict(result.get("source_chain") or {})
    source_chain.update(
        {
            "base_frozen_manifest": str(base_path),
            "base_frozen_manifest_sha256": base_sha256,
            "pair_replacement_ledger": str(ledger_path),
            "pair_replacement_ledger_sha256": ledger_sha256,
        }
    )
    result["source_chain"] = source_chain

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "g2_v2_image_instruction_mechanism_manifest.json"
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = _sha256(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(
        f"{manifest_sha256}  {destination.name}\n", encoding="utf-8"
    )
    return {
        "schema_version": G2_V2_MANIFEST_SCHEMA,
        "pair_replacement_policy": PAIR_REPLACEMENT_POLICY,
        "base_manifest": str(base_path),
        "base_manifest_sha256": base_sha256,
        "replacement_ledger": str(ledger_path),
        "replacement_ledger_sha256": ledger_sha256,
        "replacement_count": len(audit_rows),
        "case_count": len(rows),
        "execution_case_count": execution_count,
        "abstain_case_count": len(rows) - execution_count,
        "manifest": str(destination),
        "manifest_sha256": manifest_sha256,
        "digest_sidecar": str(sidecar),
        "llm_api_used": False,
        "source_asset_mutated": False,
        "target_mask_created": False,
    }


def _validate_top_level(
    base: dict[str, Any],
    ledger: dict[str, Any],
    *,
    base_sha256: str,
) -> None:
    if base.get("schema_version") != G2_V2_MANIFEST_SCHEMA:
        raise ValueError("unsupported frozen G2-v2 manifest schema")
    if base.get("primitive_ontology_version") != PRIMITIVE_ONTOLOGY_VERSION:
        raise ValueError("pair replacement requires joint primitive-v2")
    rows = base.get("cases")
    if not isinstance(rows, list) or len(rows) != int(base.get("case_count", -1)):
        raise ValueError("base frozen manifest case count is inconsistent")
    if ledger.get("schema_version") != PAIR_REPLACEMENT_LEDGER_SCHEMA:
        raise ValueError("unsupported pair replacement ledger schema")
    if ledger.get("base_manifest_sha256") != base_sha256:
        raise ValueError("pair replacement ledger is detached from the base manifest")
    if (
        ledger.get("reviewer") != "current_codex_session"
        or ledger.get("llm_api_used") is not False
    ):
        raise ValueError("pair replacements require current Codex review without API")
    replacements = ledger.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("pair replacement ledger contains no replacements")


def _materialize_replacement(
    target: dict[str, Any],
    request: dict[str, Any],
    *,
    final_case_id: str,
    replacement_ledger_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_manifest_path = Path(
        _required_text(request, "candidate_manifest")
    )
    candidate_manifest_sha256 = _required_text(
        request, "candidate_manifest_sha256"
    )
    if _sha256(candidate_manifest_path) != candidate_manifest_sha256:
        raise ValueError("candidate manifest digest drift")
    candidate_payload = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    if (
        candidate_payload.get("schema_version") != G2_V2_MANIFEST_SCHEMA
        or candidate_payload.get("primitive_ontology_version")
        != PRIMITIVE_ONTOLOGY_VERSION
    ):
        raise ValueError("replacement candidate manifest is not primitive-v2")
    candidate_case_id = _required_text(request, "candidate_case_id")
    candidates = [
        item
        for item in candidate_payload.get("cases", ())
        if str(item.get("case_id")) == candidate_case_id
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"replacement candidate identity is not unique: {candidate_case_id}"
        )
    candidate = copy.deepcopy(candidates[0])
    if not candidate.get("execution_allowed"):
        raise ValueError("replacement candidate is not enabled for exact screening")

    exact_path = Path(_required_text(request, "execution_qualification_jsonl"))
    exact_sha256 = _required_text(request, "execution_qualification_sha256")
    if _sha256(exact_path) != exact_sha256:
        raise ValueError("replacement execution qualification digest drift")
    exact_rows = [
        json.loads(line)
        for line in exact_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    exact_matches = [
        item for item in exact_rows if str(item.get("case_id")) == candidate_case_id
    ]
    if len(exact_matches) != 1:
        raise ValueError("replacement exact qualification identity is not unique")
    exact = exact_matches[0]
    if (
        exact.get("schema_version") != EXECUTION_QUALIFICATION_SCHEMA
        or exact.get("status") != "executable_preflight_passed"
        or exact.get("failure_reasons")
        or exact.get("source_manifest_sha256") != candidate_manifest_sha256
        or exact.get("llm_api_used") is not False
    ):
        raise ValueError("replacement candidate lacks a clean exact qualification")

    _validate_replacement_identity(target, candidate)
    _validate_source_assets(candidate)
    visual = request.get("visual_review")
    if not isinstance(visual, dict):
        raise ValueError("replacement is missing its visual review")
    if (
        visual.get("reviewer") != "current_codex_session"
        or visual.get("llm_api_used") is not False
        or visual.get("decision_status") != "supported_mechanism"
    ):
        raise ValueError("replacement visual review is not an approving Codex review")
    board_path = Path(_required_text(visual, "board_path"))
    board_sha256 = _required_text(visual, "board_sha256")
    if _sha256(board_path) != board_sha256:
        raise ValueError("replacement H&E board digest drift")
    observations = visual.get("visual_observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("replacement H&E review contains no observations")

    final = candidate
    final["case_id"] = final_case_id
    final["source_index"] = int(target["source_index"])
    final["seed"] = int(target["seed"])
    final["original_instruction"] = target["original_instruction"]
    final["decision_status"] = "supported_mechanism"
    final["he_decision_status"] = "supported_mechanism"
    final["decision_reason_code"] = (
        "replacement_pair_exact_and_current_codex_he_passed"
    )
    final["execution_allowed"] = True
    final["reviewed_candidate_before_execution_preflight"] = True
    final["visual_observations"] = [str(item) for item in observations]
    semantic = copy.deepcopy(final.get("prebound_semantic_intent"))
    if not isinstance(semantic, dict):
        raise ValueError("replacement candidate lacks prebound semantic intent")
    semantic["parser"] = "current_codex_session_semantic_parser_v2"
    parser_metadata = dict(semantic.get("parser_metadata") or {})
    parser_metadata.update(
        {
            "case_id": final_case_id,
            "reviewer": "current_codex_session",
            "language_authority": "current_codex_session",
            "visual_authority": True,
            "llm_api_used": False,
            "execution_runner_may_not_reparse": True,
            "pair_replacement_ledger_sha256": replacement_ledger_sha256,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "execution_qualification_sha256": exact_sha256,
        }
    )
    semantic["parser_metadata"] = parser_metadata
    if semantic.get("instruction") != final.get("instruction"):
        raise ValueError("replacement semantic intent is detached from instruction")
    if semantic.get("primitive_id") != final.get("primitive_id"):
        raise ValueError("replacement semantic intent is detached from primitive")
    final["prebound_semantic_intent"] = semantic
    final["prebound_semantic_intent_sha256"] = hashlib.sha256(
        json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    source_digests = final["source_digests"]
    final["review_basis"] = {
        "reviewer": "current_codex_session",
        "modality": "source_he_plus_tissue_and_nuclei_overlays",
        "board_path": str(board_path),
        "board_sha256": board_sha256,
        "board_page": int(visual.get("board_page", 0)),
        "board_position": int(visual.get("board_position", 0)),
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "execution_qualification_jsonl": str(exact_path),
        "execution_qualification_sha256": exact_sha256,
        "pair_replacement_ledger_sha256": replacement_ledger_sha256,
        "source_image_sha256": source_digests["image_sha256"],
        "source_tissue_mask_sha256": source_digests["tissue_mask_sha256"],
        "source_nuclei_mask_sha256": source_digests["nuclei_mask_sha256"],
        "llm_api_used": False,
    }
    final["execution_qualification"] = {
        "status": "executable_preflight_passed",
        "failure_reasons": [],
        "ledger_sha256": exact_sha256,
        "source_manifest_sha256": candidate_manifest_sha256,
    }
    metadata = dict(final.get("source_manifest_metadata") or {})
    metadata.update(
        {
            "replaces_case_id": str(target["case_id"]),
            "replacement_candidate_case_id": candidate_case_id,
            "replacement_policy": PAIR_REPLACEMENT_POLICY,
        }
    )
    final["source_manifest_metadata"] = metadata
    audit = {
        "target_case_id": str(target["case_id"]),
        "target_source_index": int(target["source_index"]),
        "final_case_id": final_case_id,
        "candidate_case_id": candidate_case_id,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "execution_qualification_sha256": exact_sha256,
        "board_sha256": board_sha256,
        "primitive_id": final["primitive_id"],
        "mechanism_id": final["mechanism_id"],
        "source_digests": copy.deepcopy(source_digests),
    }
    return final, audit


def _validate_replacement_identity(
    target: dict[str, Any], candidate: dict[str, Any]
) -> None:
    for field in (
        "organ",
        "dataset",
        "pathology_domain_id",
        "annotation_profile_id",
        "cell_observation_profile_id",
        "cell_population_profile_id",
        "legacy_primitive",
    ):
        if candidate.get(field) != target.get(field):
            raise ValueError(f"replacement changes frozen {field}")
    expected_primitive = LEGACY_PRIMITIVE_MAP.get(str(target["legacy_primitive"]))
    if expected_primitive is None or candidate.get("primitive_id") != expected_primitive:
        raise ValueError("replacement primitive differs from the legacy request")
    if not candidate.get("mechanism_id") or not candidate.get("instruction"):
        raise ValueError("replacement candidate lacks executable semantics")


def _validate_source_assets(candidate: dict[str, Any]) -> None:
    digests = candidate.get("source_digests")
    if not isinstance(digests, dict):
        raise ValueError("replacement candidate lacks source digests")
    pairs = (
        ("source_image_uri", "image_sha256"),
        ("source_tissue_mask_uri", "tissue_mask_sha256"),
        ("source_nuclei_mask_uri", "nuclei_mask_sha256"),
    )
    for path_field, digest_field in pairs:
        path = Path(_required_text(candidate, path_field))
        if not path.is_file() or _sha256(path) != digests.get(digest_field):
            raise ValueError(f"replacement source asset is missing or drifted: {path_field}")


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required text field: {key}")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

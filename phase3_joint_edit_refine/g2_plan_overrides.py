"""Apply provenance-bound current-session visual plan decisions to a shadow."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .models import JointContractError

PLAN_OVERRIDE_SCHEMA = "g2-v2-codex-visual-plan-overrides-v1"


def apply_plan_overrides(
    manifest_path: str | Path,
    overrides_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    source = Path(manifest_path)
    review = Path(overrides_path)
    rows = json.loads(source.read_text(encoding="utf-8"))
    payload = json.loads(review.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise JointContractError("shadow manifest must be a nonempty list")
    if payload.get("schema_version") != PLAN_OVERRIDE_SCHEMA:
        raise JointContractError("unsupported Codex visual plan override schema")
    reviewer = str(payload.get("reviewer") or "")
    decisions = payload.get("cases")
    if reviewer != "current_codex_session" or not isinstance(decisions, dict):
        raise JointContractError(
            "visual plan overrides require current_codex_session and case decisions"
        )
    by_id = {str(item.get("case_id")): item for item in rows}
    unknown = sorted(set(decisions) - set(by_id))
    if unknown:
        raise JointContractError(
            "visual plan overrides name cases outside the shadow: "
            + ", ".join(unknown)
        )
    source_sha = _sha256(source)
    review_sha = _sha256(review)
    result = deepcopy(rows)
    result_by_id = {str(item["case_id"]): item for item in result}
    for case_id, raw in decisions.items():
        if not isinstance(raw, dict):
            raise JointContractError(f"visual decision for {case_id} is not an object")
        anchor = raw.get("cellularity_depletion_anchor")
        _validate_depletion_anchor(case_id, anchor)
        item = result_by_id[case_id]
        if not str(item.get("primitive_id", "")).endswith("decrease-v1"):
            raise JointContractError(
                f"depletion anchor override targets non-decrease case {case_id}"
            )
        provenance = dict(item.get("provenance") or {})
        provenance["cellularity_depletion_anchor"] = deepcopy(anchor)
        provenance["codex_visual_plan_override"] = {
            "schema_version": PLAN_OVERRIDE_SCHEMA,
            "reviewer": reviewer,
            "source_manifest_sha256": source_sha,
            "override_ledger_sha256": review_sha,
            "llm_api_used": False,
        }
        item["provenance"] = provenance
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = _sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        digest + "  " + destination.name + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": PLAN_OVERRIDE_SCHEMA,
        "source_manifest": str(source),
        "source_manifest_sha256": source_sha,
        "override_ledger": str(review),
        "override_ledger_sha256": review_sha,
        "overridden_case_ids": sorted(decisions),
        "output_manifest": str(destination),
        "output_manifest_sha256": digest,
        "source_asset_mutated": False,
        "llm_api_used": False,
    }


def _validate_depletion_anchor(case_id: str, value: Any) -> None:
    if not isinstance(value, dict) or value.get("type") != "interface":
        raise JointContractError(
            f"visual decision for {case_id} lacks an interface depletion anchor"
        )
    interfaces = value.get("interface_ids")
    anchors = value.get("anchor_ids")
    observation = str(value.get("observation") or "").strip()
    confidence = float(value.get("confidence", 0.0))
    if (
        not isinstance(interfaces, list)
        or not interfaces
        or not all(isinstance(item, str) and item for item in interfaces)
        or not isinstance(anchors, list)
        or not anchors
        or not all(isinstance(item, str) and item for item in anchors)
        or not observation
        or not 0.0 <= confidence <= 1.0
    ):
        raise JointContractError(
            f"visual depletion anchor for {case_id} is incomplete"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

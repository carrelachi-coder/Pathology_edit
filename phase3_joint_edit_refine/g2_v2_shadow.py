"""Build a deterministic, stratified execution shadow from frozen G2-v2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .g2_v2_manifest import G2_V2_MANIFEST_SCHEMA

SHADOW_SELECTION_SCHEMA = "g2-v2-stratified-joint-shadow-v1"


def build_g2_v2_shadow(
    frozen_manifest_path: str | Path,
    *,
    output_dir: str | Path,
    per_organ: int = 8,
    abstain_controls: int = 2,
) -> dict[str, Any]:
    source = Path(frozen_manifest_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != G2_V2_MANIFEST_SCHEMA:
        raise ValueError("unsupported frozen G2-v2 manifest schema")
    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != int(payload.get("case_count", -1)):
        raise ValueError("frozen G2-v2 manifest case count is inconsistent")
    executable = [item for item in rows if item.get("execution_allowed")]
    rejected = [item for item in rows if not item.get("execution_allowed")]
    selected = _select_executable(executable, per_organ=per_organ)
    negative = _select_abstain_controls(rejected, count=abstain_controls)
    manifest_sha256 = _sha256(source)
    runnable = [
        _materialize_joint_context(item, manifest_sha256=manifest_sha256)
        for item in selected
    ]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runnable_path = root / "g2_v2_joint_shadow_manifest.json"
    runnable_path.write_text(
        json.dumps(runnable, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    controls_path = root / "g2_v2_abstain_controls.json"
    controls_path.write_text(
        json.dumps(negative, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection = {
        "schema_version": SHADOW_SELECTION_SCHEMA,
        "source_manifest": str(source),
        "source_manifest_sha256": manifest_sha256,
        "selection_policy": "greedy_organ_mechanism_primitive_status_coverage_v1",
        "per_organ_executable": per_organ,
        "selected_executable_count": len(selected),
        "selected_abstain_control_count": len(negative),
        "selected_case_ids": [item["case_id"] for item in selected],
        "abstain_control_case_ids": [item["case_id"] for item in negative],
        "coverage": _coverage(selected),
        "runnable_manifest": str(runnable_path),
        "runnable_manifest_sha256": _sha256(runnable_path),
        "abstain_controls": str(controls_path),
        "abstain_controls_sha256": _sha256(controls_path),
        "llm_api_used": False,
    }
    selection_path = root / "g2_v2_shadow_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**selection, "selection": str(selection_path)}


def _select_executable(rows: list[dict[str, Any]], *, per_organ: int) -> list[dict[str, Any]]:
    by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_organ[str(row["organ"])].append(row)
    selected: list[dict[str, Any]] = []
    for organ in sorted(by_organ):
        values = sorted(by_organ[organ], key=lambda item: (item["source_index"], item["case_id"]))
        if len(values) < per_organ:
            raise ValueError(f"not enough executable G2-v2 cases for {organ}")
        chosen: list[dict[str, Any]] = []
        uncovered_mechanisms = {str(item["mechanism_id"]) for item in values}
        uncovered_primitives = {str(item["primitive_id"]) for item in values}
        uncovered_statuses = {str(item["decision_status"]) for item in values}
        while len(chosen) < per_organ:
            remaining = [item for item in values if item not in chosen]
            if not remaining:
                break
            # Coverage first; then choose the case furthest from existing
            # source indices so the shadow is not a contiguous slice.
            def score(item: dict[str, Any]) -> tuple[int, int, int, int]:
                novelty = (
                    4 * (str(item["mechanism_id"]) in uncovered_mechanisms)
                    + 3 * (str(item["primitive_id"]) in uncovered_primitives)
                    + 2 * (str(item["decision_status"]) in uncovered_statuses)
                )
                distance = min(
                    (
                        abs(int(item["source_index"]) - int(current["source_index"]))
                        for current in chosen
                    ),
                    default=10_000,
                )
                return novelty, distance, -int(item["source_index"]), -len(str(item["case_id"]))

            current = max(remaining, key=score)
            chosen.append(current)
            uncovered_mechanisms.discard(str(current["mechanism_id"]))
            uncovered_primitives.discard(str(current["primitive_id"]))
            uncovered_statuses.discard(str(current["decision_status"]))
        selected.extend(chosen)
    return sorted(selected, key=lambda item: (item["organ"], item["source_index"]))


def _select_abstain_controls(
    rows: list[dict[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    chosen: list[dict[str, Any]] = []
    seen_organs: set[str] = set()
    for item in sorted(rows, key=lambda row: (row["organ"], row["source_index"])):
        organ = str(item["organ"])
        if organ in seen_organs:
            continue
        chosen.append(item)
        seen_organs.add(organ)
        if len(chosen) == count:
            return chosen
    for item in sorted(rows, key=lambda row: row["source_index"]):
        if item not in chosen:
            chosen.append(item)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise ValueError("not enough frozen abstain controls")
    return chosen


def _materialize_joint_context(
    row: dict[str, Any], *, manifest_sha256: str
) -> dict[str, Any]:
    source_digests = row["source_digests"]
    source_paths = {
        "source_image_uri": row["source_image_uri"],
        "source_tissue_mask_uri": row["source_tissue_mask_uri"],
        "source_nuclei_mask_uri": row["source_nuclei_mask_uri"],
    }
    expected = {
        "source_image_uri": source_digests["image_sha256"],
        "source_tissue_mask_uri": source_digests["tissue_mask_sha256"],
        "source_nuclei_mask_uri": source_digests["nuclei_mask_sha256"],
    }
    for key, uri in source_paths.items():
        path = Path(str(uri))
        if not path.is_file() or _sha256(path) != expected[key]:
            raise ValueError(f"frozen source asset is missing or drifted: {row['case_id']} {key}")
    metadata = dict(row.get("source_manifest_metadata") or {})
    provider = str(metadata.get("provider") or row["dataset"])
    source_site, specimen_type, primary_or_metastatic = _profile_provenance(row)
    provenance = {
        "source_image_sha256": source_digests["image_sha256"],
        "source_tissue_mask_sha256": source_digests["tissue_mask_sha256"],
        "source_nuclei_mask_sha256": source_digests["nuclei_mask_sha256"],
        "preprocessing_revision": "g2-v2-frozen-source-assets-v1",
        "original_label_map_digest": source_digests["tissue_mask_sha256"],
        "original_instance_mask_digest": source_digests["nuclei_mask_sha256"],
        "instance_authority_source": "cellvit_semantic_mask_shared_watershed_v1",
        "patch_grade": metadata.get("patch_grade", "unknown_not_recorded"),
        "provider": provider,
        "source_site": source_site,
        "specimen_type": specimen_type,
        "primary_or_metastatic": primary_or_metastatic,
        "joint_mechanism_id": row["mechanism_id"],
        "joint_primitive_id": row["primitive_id"],
        "joint_mechanism_assignment_reason": row["decision_reason_code"],
        "codex_visual_planner_observation": row["visual_observations"],
        "g2_v2_manifest_sha256": manifest_sha256,
        "g2_v2_source_index": row["source_index"],
        "g2_v2_decision_status": row["decision_status"],
        "g2_v2_review_basis": row["review_basis"],
        "require_mature_probnet_regeneration": True,
        "available_auxiliary_structures": [],
    }
    context = {
        "case_id": row["case_id"],
        "instruction": row["instruction"],
        **source_paths,
        "pathology_domain_id": row["pathology_domain_id"],
        "annotation_profile_id": row["annotation_profile_id"],
        "cell_observation_profile_id": row["cell_observation_profile_id"],
        "cell_population_profile_id": row["cell_population_profile_id"],
        "primitive_id": row["primitive_id"],
        "prebound_semantic_intent": row["prebound_semantic_intent"],
        "joint_area_budget": row.get("joint_area_budget"),
        # Scene-calibrated cell-only budgets are intentionally absent here;
        # the workflow compiles them after source instance authority exists.
        "cell_count_extent_budget": None,
        "seed": int(row["seed"]),
        "pixel_size_um": row.get("pixel_size_um"),
        "provenance": provenance,
    }
    semantic = context["prebound_semantic_intent"]
    semantic_digest = hashlib.sha256(
        json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if semantic_digest != row.get("prebound_semantic_intent_sha256"):
        raise ValueError(
            f"frozen semantic intent is missing or drifted: {row['case_id']}"
        )
    instances_uri = row.get("source_nuclei_instances_uri")
    if instances_uri:
        context["source_nuclei_instances_uri"] = instances_uri
        provenance["source_nuclei_instances_sha256"] = source_digests[
            "nuclei_instances_sha256"
        ]
    return context


def _profile_provenance(row: dict[str, Any]) -> tuple[str, str, str]:
    dataset = str(row["dataset"])
    image = str(row["source_image_uri"]).casefold()
    if dataset == "IGNITE":
        return "lung", "resection", "not_applicable"
    if dataset == "PUMA":
        if "metastatic" in image:
            return "metastatic_site_not_specified", "unknown_not_recorded", "metastatic"
        if "primary" in image:
            return "skin", "unknown_not_recorded", "primary"
        return "unknown_not_recorded", "unknown_not_recorded", "unknown_not_recorded"
    return "unknown_not_recorded", "unknown_not_recorded", "not_applicable"


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_organ": dict(sorted(Counter(str(item["organ"]) for item in rows).items())),
        "by_status": dict(sorted(Counter(str(item["decision_status"]) for item in rows).items())),
        "by_primitive": dict(sorted(Counter(str(item["primitive_id"]) for item in rows).items())),
        "by_mechanism": dict(sorted(Counter(str(item["mechanism_id"]) for item in rows).items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

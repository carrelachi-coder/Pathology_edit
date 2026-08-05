"""Build a reproducible, stratified G2 joint-condition pilot manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_G2_PRIMITIVES = {
    "tumor_increase": "tumor-burden-increase-v1",
    "tumor_decrease": "tumor-burden-decrease-v1",
    "stroma_increase": "stroma-increase-v1",
}

ORGAN_CONTRACTS = {
    "breast": ("breast-invasive-carcinoma-v1", "bcss-semantic-v1", "breast-cellvit-source-first-v1"),
    "colorectal": ("colorectal-adenocarcinoma-v1", "glas-gland-v1", "colorectal-cellvit-source-first-v1"),
    "prostate": ("prostate-adenocarcinoma-v1", "panda-gleason-v1", "prostate-cellvit-source-first-v1"),
    "lung": ("lung-carcinoma-v1", "ignite-semantic-v1", "lung-cellvit-source-first-v1"),
    "oral": ("oral-squamous-cell-carcinoma-v1", "orca-semantic-v1", "oral-scc-cellvit-source-first-v1"),
    "skin": ("melanoma-v1", "puma-semantic-v1", "melanoma-cellvit-source-first-v1"),
}

DEFAULT_RESEARCH_MECHANISMS = {
    "breast": "breast-cohesive-nst-front",
    "colorectal": "colorectal-gland-forming-front",
    "prostate": "prostate-pattern-5-growth",
    "lung": "lung-solid-squamous-growth",
    "oral": "oral-scc-dispersed-invasive-front",
    "skin": "melanoma-cohesive-nest-sheet",
}


def select_stratified_cases(payload: dict, *, per_organ: dict[str, int], mandatory_case_ids=()) -> list[dict]:
    rows = [
        item for item in payload["cases"]
        if item.get("organ") in per_organ and item.get("g2_primitive") in SUPPORTED_G2_PRIMITIVES
    ]
    by_id = {item["case_id"]: item for item in rows}
    selected = []
    selected_ids = set()
    for case_id in mandatory_case_ids:
        if case_id in by_id:
            selected.append(by_id[case_id])
            selected_ids.add(case_id)
    grouped = defaultdict(lambda: defaultdict(list))
    for item in rows:
        if item["case_id"] not in selected_ids:
            grouped[item["organ"]][item["g2_primitive"]].append(item)
    for organ in grouped:
        for primitive in grouped[organ]:
            grouped[organ][primitive].sort(key=lambda item: item["case_id"])
    existing = Counter(item["organ"] for item in selected)
    for organ, quota in per_organ.items():
        primitives = sorted(grouped[organ])
        cursor = Counter()
        while existing[organ] < quota:
            progressed = False
            for primitive in primitives:
                index = cursor[primitive]
                values = grouped[organ][primitive]
                if index >= len(values):
                    continue
                selected.append(values[index])
                cursor[primitive] += 1
                existing[organ] += 1
                progressed = True
                if existing[organ] >= quota:
                    break
            if not progressed:
                raise ValueError(f"not enough supported G2 cases for {organ}")
    return sorted(selected, key=lambda item: item["case_id"])


def write_fetch_plan(rows: list[dict], *, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selection = output / "g2_joint_pilot_selection.json"
    selection.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    files = sorted({str(item[key]) for item in rows for key in ("source_image", "source_tissue_mask", "source_nuclei_mask")})
    fetch = output / "g2_joint_pilot_fetch_files.txt"
    fetch.write_text("\n".join(path.lstrip("/") for path in files) + "\n", encoding="utf-8")
    return {"selection": str(selection), "fetch_files": str(fetch)}


def build_local_joint_records(
    rows: list[dict],
    *,
    asset_root: str | Path,
    mechanism_decisions: dict[str, str] | None = None,
) -> list[dict]:
    asset_root = Path(asset_root)
    mechanism_decisions = mechanism_decisions or {}
    records = []
    for row in rows:
        organ = row["organ"]
        domain, annotation, population = ORGAN_CONTRACTS[organ]
        local = {
            key: asset_root / str(row[key]).lstrip("/")
            for key in ("source_image", "source_tissue_mask", "source_nuclei_mask")
        }
        missing = [str(path) for path in local.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("pilot assets are missing: " + ", ".join(missing))
        tissue_digest = _sha256(local["source_tissue_mask"])
        provenance = {
            "source_image_sha256": _sha256(local["source_image"]),
            "source_tissue_mask_sha256": tissue_digest,
            "source_nuclei_mask_sha256": _sha256(local["source_nuclei_mask"]),
            "preprocessing_revision": "g2-frozen-product-manifest-source",
            "original_label_map_digest": tissue_digest,
            "original_instance_mask_digest": tissue_digest,
            "patch_grade": row.get("patch_grade", "unknown_not_recorded"),
            "provider": row.get("provider", row.get("dataset", "unknown_not_recorded")),
            "source_site": row.get("source_site", "unknown_not_recorded"),
            "specimen_type": row.get("specimen_type", "unknown_not_recorded"),
            "primary_or_metastatic": row.get("primary_or_metastatic", "unknown_not_recorded"),
            "joint_mechanism_id": mechanism_decisions.get(row["case_id"], DEFAULT_RESEARCH_MECHANISMS[organ]),
            "available_auxiliary_structures": row.get("available_auxiliary_structures", []),
            "g2_source_case_id": row["case_id"],
            "g2_source_mask_annotation_provenance": row.get("source_mask_annotation_provenance"),
        }
        records.append(
            {
                "case_id": row["case_id"],
                "instruction": row["instruction"],
                "source_image_uri": str(local["source_image"]),
                "source_tissue_mask_uri": str(local["source_tissue_mask"]),
                "source_nuclei_mask_uri": str(local["source_nuclei_mask"]),
                "pathology_domain_id": domain,
                "annotation_profile_id": annotation,
                "cell_observation_profile_id": "cellvit-five-class-v1",
                "cell_population_profile_id": population,
                "primitive_id": SUPPORTED_G2_PRIMITIVES[row["g2_primitive"]],
                "joint_area_budget": {
                    "target_fraction": 0.19,
                    "min_fraction": 0.14,
                    "max_fraction": 0.24,
                    "tissue_min_fraction": 0.14,
                    "relative_tolerance": 0.02,
                    "fallback_policy": "max_feasible_below_target",
                    "capacity_floor_policy": "lower_to_proven_max_safe",
                },
                "seed": int(row.get("organic_seed", 42)),
                "pixel_size_um": row.get("pixel_size_um"),
                "provenance": provenance,
            }
        )
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

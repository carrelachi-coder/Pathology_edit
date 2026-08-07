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
    "breast": {
        "tumor_increase": "breast-cohesive-nst-front",
        "tumor_decrease": "breast-cohesive-nst-front",
        "stroma_increase": "breast-cohesive-nst-front",
    },
    "colorectal": {
        "tumor_increase": "colorectal-gland-forming-front",
        "tumor_decrease": "colorectal-gland-forming-front",
        "stroma_increase": "colorectal-gland-forming-front",
    },
    "prostate": {
        "tumor_increase": "prostate-pattern-5-growth",
        "tumor_decrease": "prostate-pattern-5-growth",
        "stroma_increase": "prostate-pattern-5-growth",
    },
    "lung": {
        "tumor_increase": "lung-solid-squamous-growth",
        "tumor_decrease": "lung-solid-squamous-growth",
        "stroma_increase": "lung-solid-squamous-growth",
    },
    # The dispersed-front mechanism is cell-only and must never be silently
    # assigned to a tissue-burden primitive.
    "oral": {
        "tumor_increase": "oral-scc-cohesive-nest-cord",
        "tumor_decrease": "oral-scc-cohesive-nest-cord",
        "stroma_increase": "oral-scc-cohesive-nest-cord",
    },
    "skin": {
        "tumor_increase": "melanoma-cohesive-nest-sheet",
        "tumor_decrease": "melanoma-cohesive-nest-sheet",
        "stroma_increase": "melanoma-cohesive-nest-sheet",
    },
}

OPTIONAL_ASSET_KEYS = (
    "source_nuclei_instances",
    "source_nuclei_instances_uri",
)


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
    files = {
        str(item[key])
        for item in rows
        for key in ("source_image", "source_tissue_mask", "source_nuclei_mask")
    }
    for item in rows:
        for key in OPTIONAL_ASSET_KEYS:
            if item.get(key):
                files.add(str(item[key]))
        auxiliary = item.get("auxiliary_structure_uris", {})
        if isinstance(auxiliary, dict):
            files.update(str(value) for value in auxiliary.values())
    files = sorted(files)
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
        instance_source = next(
            (row.get(key) for key in OPTIONAL_ASSET_KEYS if row.get(key)),
            None,
        )
        local_instances = (
            asset_root / str(instance_source).lstrip("/")
            if instance_source
            else None
        )
        if local_instances is not None and not local_instances.is_file():
            raise FileNotFoundError(
                "pilot native nucleus instances are missing: "
                + str(local_instances)
            )
        raw_auxiliary = row.get("auxiliary_structure_uris", {})
        if raw_auxiliary is None:
            raw_auxiliary = {}
        if not isinstance(raw_auxiliary, dict):
            raise TypeError("auxiliary_structure_uris must be a mapping")
        local_auxiliary = {
            str(structure_id): asset_root / str(path).lstrip("/")
            for structure_id, path in raw_auxiliary.items()
        }
        missing_auxiliary = [
            str(path) for path in local_auxiliary.values() if not path.is_file()
        ]
        if missing_auxiliary:
            raise FileNotFoundError(
                "pilot auxiliary structures are missing: "
                + ", ".join(missing_auxiliary)
            )
        tissue_digest = _sha256(local["source_tissue_mask"])
        mechanism, mechanism_reason = _resolve_mechanism(
            row,
            mechanism_decisions=mechanism_decisions,
            local_auxiliary=local_auxiliary,
            local_instances=local_instances,
        )
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
            "joint_mechanism_id": mechanism,
            "joint_mechanism_assignment_reason": mechanism_reason,
            "require_mature_probnet_regeneration": True,
            "available_auxiliary_structures": sorted(local_auxiliary),
            "g2_source_case_id": row["case_id"],
            "g2_source_mask_annotation_provenance": row.get("source_mask_annotation_provenance"),
        }
        if local_instances is not None:
            provenance["source_nuclei_instances_sha256"] = _sha256(
                local_instances
            )
        for structure_id, path in local_auxiliary.items():
            provenance[f"auxiliary_structure_sha256.{structure_id}"] = _sha256(
                path
            )
        record = {
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
                    # G2 burden edits may fall back from the 19% target only
                    # to the largest proven-safe realization that still meets
                    # the agreed 14% meaningful tissue floor.  A smaller
                    # capacity is an auditable abstention, not a successful
                    # visually negligible edit.
                    "minimum_effective_fraction": 0.14,
                },
                "seed": int(row.get("organic_seed", 42)),
                "pixel_size_um": row.get("pixel_size_um"),
                "provenance": provenance,
            }
        if local_instances is not None:
            record["source_nuclei_instances_uri"] = str(local_instances)
        if local_auxiliary:
            record["auxiliary_structure_uris"] = {
                key: str(path) for key, path in sorted(local_auxiliary.items())
            }
        records.append(record)
    return records


def _resolve_mechanism(
    row: dict,
    *,
    mechanism_decisions: dict[str, str],
    local_auxiliary: dict[str, Path],
    local_instances: Path | None,
) -> tuple[str, str]:
    explicit = mechanism_decisions.get(row["case_id"])
    if explicit:
        return explicit, "explicit_visual_planner_decision"
    organ = str(row["organ"])
    primitive = str(row["g2_primitive"])
    mechanism = DEFAULT_RESEARCH_MECHANISMS[organ][primitive]
    if mechanism == "colorectal-gland-forming-front":
        missing = []
        if "gland_or_lumen_support" not in local_auxiliary:
            missing.append("gland_or_lumen_support")
        if local_instances is None:
            missing.append("source_nuclei_instances")
        if missing:
            return (
                "__abstain__",
                "default colorectal gland mechanism lacks required assets: "
                + ",".join(missing),
            )
    return mechanism, "primitive_compatible_research_default"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

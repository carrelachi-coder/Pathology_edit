#!/usr/bin/env python3
"""Freeze an independent six-dataset source-qualified D/E mask-edit cohort.

Selection uses source masks and recorded provenance only. It never consults
Planner, executor, gate, or E-audit outcomes for an individual candidate.
The capability and quota choices are disclosed in the output protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path("/data1/lyw/pathology_edit_eval/r2_de_terra_six_v4_20260924")
PREVIOUS = Path("/data1/lyw/pathology_edit_eval/r2_de_terra_six_v3_20260924")
CODE = ROOT / "code"
META = Path("/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_val.json")
DATA = Path("/data/wqx/flowedit/data")
PROBNET = Path("/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt")

# Preset allocation: 37 requests per dataset, with tissue and cell edits.
QUOTAS = (
    ("BCSS", "cohesive-boundary-expansion-v1", 27),
    ("BCSS", "generic-immune-infiltrate-decrease-v1", 10),
    ("GLAS", "cellularity-decrease-v1", 37),
    ("ORCA", "cellularity-decrease-v1", 37),
    ("PANDA", "cell-type-abundance-decrease-v1", 10),
    ("PANDA", "cohesive-boundary-expansion-v1", 27),
    ("IGNITE", "cell-type-abundance-decrease-v1", 27),
    ("IGNITE", "cohesive-boundary-expansion-v1", 10),
    ("PUMA", "cell-type-abundance-decrease-v1", 8),
    ("PUMA", "cohesive-boundary-expansion-v1", 9),
    ("PUMA", "cell-type-abundance-increase-v1", 15),
    ("PUMA", "neoplastic-cell-abundance-decrease-v1", 5),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sample_group(dataset: str, sample_id: str, row: dict) -> str:
    if dataset == "BCSS":
        return sample_id.split("_x")[0]
    if dataset == "PANDA":
        return sample_id.split("_y")[0]
    if dataset in {"GLAS", "IGNITE", "PUMA"}:
        return re.split(r"_py|_px", sample_id)[0]
    return str(row.get("case_id") or re.split(r"_py|_px", sample_id)[0])


def patch_metadata(dataset: str) -> dict[str, tuple[str, str]]:
    path = DATA / f"{dataset if dataset != 'GLAS' else 'GlaS'}_PATCHES" / "metadata.jsonl"
    if not path.is_file():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row["image"].replace("\\", "/").split("/")[-1]
        records[Path(name).stem] = (str(row.get("text", "")), hashlib.sha256(line.encode()).hexdigest())
    return records


def source_screen(dataset: str, primitive: str, row: dict, text: str) -> tuple[bool, str, dict]:
    image = Path(row["target_image"])
    tissue_path = Path(row["target_tissue_mask"])
    nuclei_path = Path(row["target_nuclei_mask"])
    if not all(p.is_file() for p in (image, tissue_path, nuclei_path)):
        return False, "source_file_missing", {}
    tissue = np.asarray(Image.open(tissue_path))
    nuclei = np.asarray(Image.open(nuclei_path))
    if tissue.ndim != 2 or nuclei.shape != tissue.shape or Image.open(image).size != tissue.shape[::-1]:
        return False, "source_shape_mismatch", {}
    sample = row["sample_id"]
    group = sample_group(dataset, sample, row)
    facts = {"sample_id": sample, "source_group": group}

    if dataset == "PUMA":
        # The dataset records primary versus metastatic; exact metastatic
        # anatomic sites are unavailable in the local release.
        if "_primary_" not in sample or "primary melanoma" not in text:
            return False, "puma_primary_provenance_required", facts
        facts.update(primary_or_metastatic="primary", source_site="skin",
                     source_site_evidence="PUMA primary cutaneous melanoma cohort; exact subsite unavailable")
    elif dataset == "IGNITE":
        if "resection from lung" in text:
            specimen = "resection"
        elif "biopsy from lung" in text:
            specimen = "biopsy"
        else:
            return False, "ignite_source_text_missing_site_or_specimen", facts
        facts.update(source_site="lung", specimen_type=specimen,
                     source_site_evidence="exact patch entry in IGNITE metadata.jsonl")
    elif dataset == "PANDA":
        # This establishes the provider of the *annotation protocol*, not a
        # verified slide-acquisition center. Do not claim institutional origin.
        if not np.isin(tissue, (8, 9, 10)).any():
            return False, "panda_gleason_fine_labels_absent", facts
        facts.update(provider="radboud_style_gleason_label_schema",
                     provider_evidence="original Gleason-pattern label mapping plus source fine IDs 8/9/10; acquisition center unverified")

    if primitive == "cell-type-abundance-increase-v1":
        if dataset != "PUMA":
            return False, "cell_increase_profile_not_in_v4", facts
        qualified = _complete_instance_zones(tissue, nuclei, class_id=2)
        reference_count = sum(zone["complete_count"] for zone in qualified)
        facts["complete_inflammatory_reference_instances"] = reference_count
        free_stroma = int(np.count_nonzero((tissue == 2) & (nuclei == 0)))
        facts["free_stroma_pixels"] = free_stroma
        if reference_count < 8 or free_stroma < 15000:
            return False, "cell_increase_reference_or_host_space_below_screen", facts
    elif primitive == "neoplastic-cell-abundance-decrease-v1":
        if dataset != "PUMA":
            return False, "neoplastic_decrease_profile_not_in_v4", facts
        qualified = _complete_instance_zones(
            tissue, nuclei, class_id=1, allowed_fine_ids={1}
        )
        viable = [zone for zone in qualified if zone["complete_count"] >= 24
                  and zone["center_span_px"] >= 80]
        facts["max_local_complete_neoplastic_instances"] = max(
            (zone["complete_count"] for zone in qualified), default=0
        )
        if not viable:
            return False, "neoplastic_local_instance_capacity_below_screen", facts
    elif primitive == "cell-type-abundance-decrease-v1":
        labels, _ = ndimage.label(nuclei == 102)
        complete_proxy = int(np.count_nonzero(np.bincount(labels.ravel())[1:] >= 20))
        floor = {"PANDA": 12, "IGNITE": 40, "PUMA": 30}[dataset]
        facts["inflammatory_component_proxy_ge20px"] = complete_proxy
        if complete_proxy < floor:
            return False, "inflammatory_capacity_proxy_below_floor", facts
        qualified = _complete_instance_zones(
            tissue, nuclei, class_id=2,
            allowed_fine_ids={2} if dataset == "PANDA" else None,
        )
        minimum = {"PANDA": 8, "IGNITE": 24, "PUMA": 16}[dataset]
        viable = [zone for zone in qualified if zone["complete_count"] >= minimum
                  and zone["center_span_px"] >= 80]
        facts["max_local_complete_inflammatory_instances"] = max(
            (zone["complete_count"] for zone in qualified), default=0
        )
        facts["max_local_inflammatory_center_span_px"] = max(
            (zone["center_span_px"] for zone in qualified), default=0
        )
        if not viable:
            return False, "inflammatory_local_instance_capacity_below_screen", facts
    elif primitive == "cellularity-decrease-v1":
        total = int(np.count_nonzero(nuclei))
        facts["nuclear_foreground_pixels"] = total
        if total < 6000:
            return False, "cellularity_source_below_6000px", facts
        qualified = _complete_instance_zones(tissue, nuclei, class_id=None)
        viable = [zone for zone in qualified if zone["complete_count"] >= 48
                  and zone["center_span_px"] >= 120]
        facts["max_local_complete_nuclei"] = max(
            (zone["complete_count"] for zone in qualified), default=0
        )
        if not viable:
            return False, "cellularity_local_instance_capacity_below_screen", facts
    elif primitive == "generic-immune-infiltrate-decrease-v1":
        immune = int(np.count_nonzero(tissue == 4))
        facts["immune_tissue_pixels"] = immune
        if immune < 4096:
            return False, "immune_compartment_below_4096px", facts
        if immune < 12000:
            return False, "immune_compartment_below_12000px", facts
    elif primitive == "cohesive-boundary-expansion-v1":
        tumor = np.isin(tissue, (8, 9, 10)) if dataset == "PANDA" else tissue == 1
        stroma = tissue == 2
        front = ndimage.binary_dilation(tumor, structure=np.ones((3, 3))) & stroma
        facts.update(tumor_pixels=int(tumor.sum()), stroma_pixels=int(stroma.sum()),
                     external_front_pixels=int(front.sum()), neoplastic_nuclear_pixels=int(np.count_nonzero(nuclei == 101)))
        if not (0.05 <= tumor.mean() <= 0.75 and stroma.mean() >= 0.12
                and front.sum() >= 200 and np.count_nonzero(nuclei == 101) >= 500):
            return False, "tissue_source_or_external_front_insufficient", facts
        if dataset == "PANDA":
            # Pattern 3 alone cannot authorize the two active growth skills.
            pattern = next((fine for fine in (9, 10)
                            if np.count_nonzero(
                                ndimage.binary_dilation(tissue == fine,
                                                        structure=np.ones((3, 3)))
                                & stroma) >= 350), None)
            if pattern is None:
                return False, "panda_pattern4_or_5_external_front_absent", facts
            facts["requested_gleason_fine_id"] = pattern
        target_front = (
            tissue == facts["requested_gleason_fine_id"]
            if dataset == "PANDA" else tumor
        )
        nearby_stroma = stroma & (
            ndimage.distance_transform_edt(~target_front) <= 20
        )
        facts["stroma_within_20px_of_target_front"] = int(nearby_stroma.sum())
        if nearby_stroma.sum() < 6816:
            return False, "bounded_external_growth_capacity_below_floor", facts
    else:
        raise ValueError(primitive)
    return True, "source_screen_passed", facts


def _complete_instance_zones(
    tissue: np.ndarray, nuclei: np.ndarray, *, class_id: int | None,
    allowed_fine_ids: set[int] | None = None,
) -> list[dict]:
    """Source-only complete-instance screen, before any Planner or edit outcome.

    This is deliberately a necessary-condition screen. It does not certify a
    depletion gradient, a cell program, or a successful mask edit.
    """
    from phase3_joint_edit_refine.nuclei import iter_instances

    components: dict[int, np.ndarray] = {}
    zones: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for _instance_id, observed_class, footprint in iter_instances(nuclei):
        if class_id is not None and observed_class != class_id:
            continue
        rows, cols = np.nonzero(footprint)
        if not len(rows) or min(rows.min(), cols.min()) == 0 or (
            rows.max() == tissue.shape[0] - 1
            or cols.max() == tissue.shape[1] - 1
        ):
            continue
        fine_ids = np.unique(tissue[footprint])
        if len(fine_ids) != 1 or fine_ids[0] == 0:
            continue
        fine_id = int(fine_ids[0])
        if allowed_fine_ids is not None and fine_id not in allowed_fine_ids:
            continue
        if fine_id not in components:
            components[fine_id], _ = ndimage.label(tissue == fine_id)
        center_row, center_col = int(np.median(rows)), int(np.median(cols))
        component_id = int(components[fine_id][center_row, center_col])
        if component_id == 0 or not np.all(
            components[fine_id][footprint] == component_id
        ):
            continue
        zones[(fine_id, component_id)].append((float(rows.mean()), float(cols.mean())))
    result = []
    for (fine_id, component_id), centers in zones.items():
        points = np.asarray(centers)
        span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        result.append({"fine_id": fine_id, "component_id": component_id,
                       "complete_count": len(centers),
                       "center_span_px": round(span, 2)})
    return result


def main() -> None:
    if (ROOT / "frozen_cohort.json").exists():
        raise RuntimeError("refusing to overwrite frozen cohort")
    previous = json.loads((PREVIOUS / "frozen_cohort.json").read_text())
    previous_samples = {
        (item["dataset"], item["sample_id"]) for item in previous
    }
    sys.path.insert(0, str(CODE))
    benchmark = [json.loads(line) for line in (CODE / "benchmarks/semantic_parser_planner_v1/benchmark.jsonl").read_text().splitlines()]
    bindings = {}
    for item in benchmark:
        if (item["category"] == "catalog_single_intent" and item["language"] == "en"
                and item["gold_semantic_request"]["intents"][0]["strength"] == "unspecified"):
            bindings[(item["case_profile"]["dataset"].upper(), item["catalog_target_primitive_ids"][0])] = item
    metadata = json.loads(META.read_text())["pairs"]
    pools = defaultdict(dict)
    for item in metadata:
        pools[item["dataset"].upper()].setdefault(item["sample_id"], item)
    patch_text = {ds: patch_metadata(ds) for ds in {row[0] for row in QUOTAS}}
    used_samples = defaultdict(Counter)
    used_groups = defaultdict(Counter)
    cohort = []
    selection = []
    for dataset, primitive, quota in QUOTAS:
        template = bindings[(dataset, primitive)]
        candidates = sorted(pools[dataset].values(), key=lambda row: hashlib.sha256(
            f"r2-de-six-v4|{dataset}|{primitive}|{row['sample_id']}".encode()).hexdigest())
        exclusions = Counter()
        chosen = []
        for source in candidates:
            sample = source["sample_id"]
            if (dataset, sample) in previous_samples:
                exclusions["v3_patch_excluded"] += 1
                continue
            maximum_uses = 2 if dataset == "PUMA" else 1
            if used_samples[dataset][sample] >= maximum_uses:
                exclusions["per_patch_request_cap"] += 1
                continue
            group = sample_group(dataset, sample, source)
            if used_groups[dataset][group] >= 8:
                exclusions["source_group_cap_8"] += 1
                continue
            source_text, metadata_line_sha = patch_text[dataset].get(sample, ("", ""))
            passed, reason, facts = source_screen(dataset, primitive, source, source_text)
            if not passed:
                exclusions[reason] += 1
                continue
            used_samples[dataset][sample] += 1
            used_groups[dataset][group] += 1
            chosen.append((source, facts, metadata_line_sha))
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise RuntimeError(f"{dataset}/{primitive}: selected {len(chosen)} of {quota}; exclusions={dict(exclusions)}")
        selection.append({"dataset": dataset, "primitive_id": primitive, "quota": quota,
                          "selected_samples": [row[0]["sample_id"] for row in chosen],
                          "exclusions": dict(exclusions)})
        for source, facts, line_sha in chosen:
            profile = template["case_profile"]
            sample = source["sample_id"]
            case_id = f"r2dev4-{len(cohort)+1:03d}-{dataset.lower()}-{primitive.removesuffix('-v1')}"
            paths = {"source_image_uri": source["target_image"],
                     "source_tissue_mask_uri": source["target_tissue_mask"],
                     "source_nuclei_mask_uri": source["target_nuclei_mask"]}
            digests = {name: sha(Path(path)) for name, path in paths.items()}
            source_provenance = {
                "source_image_sha256": digests["source_image_uri"],
                "source_tissue_mask_sha256": digests["source_tissue_mask_uri"],
                "source_nuclei_mask_sha256": digests["source_nuclei_mask_uri"],
                "original_label_map_digest": digests["source_tissue_mask_uri"],
                "original_instance_mask_digest": digests["source_nuclei_mask_uri"],
                "preprocessing_revision": "metadata_cross_val-existing-assets-readonly-r2de-six-v4",
                "require_mature_probnet_regeneration": True,
                "require_frozen_probnet_spatial_ranker": True,
                "instance_authority_source": "cellvit_semantic_mask_shared_watershed_v1",
                "source_metadata_sha256": sha(META),
                "dataset_patch_metadata_line_sha256": line_sha,
                "r2_source_group": facts["source_group"],
                "r2_group_authority": "documented_filename_group_not_verified_patient",
                "r2_selection": "six_dataset_v4_source_mask_instance_and_recorded_metadata_only",
                "r2_expected_primitive": primitive,
            }
            source_provenance.update({k: v for k, v in facts.items() if k not in {"sample_id", "source_group"}})
            request = dict(template["gold_semantic_request"])
            request.pop("request_sha256", None)
            instruction = template["instruction"]
            if dataset == "PANDA" and primitive == "cohesive-boundary-expansion-v1":
                pattern = 4 if facts["requested_gleason_fine_id"] == 9 else 5
                instruction = (
                    f"Expand the existing Gleason pattern {pattern} tumor "
                    "boundary cohesively while preserving other fine patterns"
                )
                request["instruction"] = instruction
                request["intents"] = [
                    {**item, "source_text": instruction}
                    for item in request["intents"]
                ]
            cell_only = primitive != "cohesive-boundary-expansion-v1" and primitive != "generic-immune-infiltrate-decrease-v1"
            record = {
                "case_id": case_id,
                "instruction": instruction,
                **{k: profile[k] for k in ("pathology_domain_id", "annotation_profile_id",
                                                 "cell_observation_profile_id", "cell_population_profile_id")},
                **paths,
                "source_nuclei_instances_uri": None,
                "auxiliary_structure_uris": {},
                "pixel_size_um": None,
                "seed": 42,
                "joint_area_budget": None if cell_only else {
                    "target_fraction": .04, "min_fraction": .026, "max_fraction": .06,
                    "tissue_min_fraction": .026, "relative_tolerance": .05,
                    "fallback_policy": "max_feasible_below_target", "capacity_floor_policy": "strict",
                },
                "cell_count_extent_budget": None,
                "provenance": source_provenance,
                "prebound_semantic_request": request,
            }
            cohort.append({"index": len(cohort), "dataset": dataset, "primitive_id": primitive,
                           "source_group": facts["source_group"], "sample_id": sample,
                           "level": "cell" if cell_only else "tissue", "record": record})
    if len(cohort) != 222 or {row["dataset"] for row in cohort} != {"BCSS", "GLAS", "ORCA", "PANDA", "IGNITE", "PUMA"}:
        raise RuntimeError("six-dataset 222-request allocation mismatch")
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "frozen_cohort.json", cohort)
    write_json(ROOT / "selection_audit.json", selection)
    protocol = {
        "version": "r2-de-six-v4",
        "runtime_code_base_commit": "2db8e23",
        "builder_sha256": sha(Path(__file__)),
        "cohort_sha256": sha(ROOT / "frozen_cohort.json"),
        "source_metadata_sha256": sha(META),
        "requests": len(cohort),
        "per_dataset": dict(Counter(row["dataset"] for row in cohort)),
        "per_binding": {f"{ds}/{primitive}": n for ds, primitive, n in QUOTAS},
        "source_groups_per_dataset": {ds: len(groups) for ds, groups in used_groups.items()},
        "case_selection": "Preset six-dataset quotas; exclude all v3 patches; source-only mask and complete-instance screens before freeze; PUMA may have two distinct primitives on one patch; no Planner, executor, gate or E outcome selection",
        "capability_selection": "Bindings and thresholds chosen after v3 audit, so this is a source-qualified follow-up benchmark, not a replacement for v3 or an all-primitive estimate",
        "v3_cohort_sha256": sha(PREVIOUS / "frozen_cohort.json"),
        "panda_provider_scope": "Radboud-style annotation schema inferred from source fine Gleason IDs; acquisition institution not independently verified",
        "puma_site_scope": "Primary cutaneous melanoma at dataset level; exact anatomic subsite unavailable",
        "grouping_scope": "Filename-derived source groups; patches within a group are correlated and are not separate patients",
        "parser_mode": "prebound benchmark gold semantics; parser accuracy excluded",
        "planner_mode": "fresh gpt-5.6-terra Codex CLI sessions",
        "execution_mode": "mature ProbNet frozen epoch29, meta-eval",
        "per_request_timeout_seconds": 600,
        "probnet_checkpoint_sha256": sha(PROBNET),
        "gpu_ids": [0, 2, 6, 7],
        "outcome_selection": "No replacement or reroll after freeze",
    }
    write_json(ROOT / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()

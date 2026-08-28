#!/usr/bin/env python3
"""Build the bilingual v4 Parser/Planner conformance benchmark.

The benchmark is catalog-derived and deterministic.  It is intended for
interface regression testing, not as evidence of unrestricted clinical
language understanding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.models import JointAreaBudget, JointCaseContext
from phase3_joint_edit_refine.program_planner import SemanticProgramPlanner
from phase3_joint_edit_refine.semantic_request import (
    SEMANTIC_REQUEST_SCHEMA_VERSION,
    IntentRelation,
    SemanticIntentClause,
    SemanticRequest,
)
from phase3_joint_edit_refine.skills.repository import JointSkillRepository

BENCHMARK_VERSION = "semantic-parser-planner-benchmark-v1"
DEFAULT_OUTPUT_DIR = Path("benchmarks/semantic_parser_planner_v1")


@dataclass(frozen=True)
class ProfileSpec:
    dataset: str
    organ: str
    pathology_domain_id: str
    annotation_profile_id: str
    cell_population_profile_id: str


@dataclass(frozen=True)
class PrimitiveSpec:
    primitive_id: str
    target: str
    operation: str
    spatial_scope: str = "unspecified"
    morphology: str = "unspecified"
    cell_class: str | None = None
    clinical_context: str = "none"
    zh: str = ""
    en: str = ""


PROFILES = (
    ProfileSpec(
        "BCSS",
        "breast",
        "breast-invasive-carcinoma-v1",
        "bcss-semantic-v1",
        "breast-cellvit-source-first-v1",
    ),
    ProfileSpec(
        "PANDA",
        "prostate",
        "prostate-adenocarcinoma-v1",
        "panda-gleason-v1",
        "prostate-cellvit-source-first-v1",
    ),
    ProfileSpec(
        "GLaS",
        "colorectal",
        "colorectal-adenocarcinoma-v1",
        "glas-gland-v1",
        "colorectal-cellvit-source-first-v1",
    ),
    ProfileSpec(
        "IGNITE",
        "lung",
        "lung-carcinoma-v1",
        "ignite-semantic-v1",
        "lung-cellvit-source-first-v1",
    ),
    ProfileSpec(
        "ORCA",
        "oral",
        "oral-squamous-cell-carcinoma-v1",
        "orca-semantic-v1",
        "oral-scc-cellvit-source-first-v1",
    ),
    ProfileSpec(
        "PUMA",
        "skin",
        "melanoma-v1",
        "puma-semantic-v1",
        "melanoma-cellvit-source-first-v1",
    ),
)


PRIMITIVE_SPECS = {
    item.primitive_id: item
    for item in (
        PrimitiveSpec(
            "cell-type-abundance-increase-v1",
            "selected_cell_population",
            "increase",
            spatial_scope="local",
            cell_class="inflammatory",
            zh="在局部增加炎症细胞",
            en="Increase inflammatory cells locally",
        ),
        PrimitiveSpec(
            "cell-type-abundance-decrease-v1",
            "selected_cell_population",
            "decrease",
            spatial_scope="local",
            cell_class="inflammatory",
            zh="在局部减少炎症细胞",
            en="Decrease inflammatory cells locally",
        ),
        PrimitiveSpec(
            "cellularity-increase-v1",
            "overall_cellularity",
            "increase",
            spatial_scope="local",
            zh="增加局部总体细胞密度",
            en="Increase overall cellularity locally",
        ),
        PrimitiveSpec(
            "cellularity-decrease-v1",
            "overall_cellularity",
            "decrease",
            spatial_scope="local",
            zh="降低局部总体细胞密度",
            en="Decrease overall cellularity locally",
        ),
        PrimitiveSpec(
            "cohesive-boundary-expansion-v1",
            "tumor_extent",
            "increase",
            spatial_scope="boundary",
            morphology="cohesive",
            zh="让肿瘤边界连续地向外扩张",
            en="Expand the tumor boundary cohesively",
        ),
        PrimitiveSpec(
            "generic-immune-infiltrate-increase-v1",
            "immune_compartment",
            "increase",
            spatial_scope="peritumoral",
            zh="在瘤周增加免疫浸润区域",
            en="Increase the peritumoral immune compartment",
        ),
        PrimitiveSpec(
            "generic-immune-infiltrate-decrease-v1",
            "immune_compartment",
            "decrease",
            spatial_scope="peritumoral",
            zh="在瘤周减少免疫浸润区域",
            en="Decrease the peritumoral immune compartment",
        ),
        PrimitiveSpec(
            "invasive-cord-formation-v1",
            "invasion_pattern",
            "increase",
            spatial_scope="peritumoral",
            morphology="cord",
            zh="在瘤周形成浸润性肿瘤条索",
            en="Form invasive tumor cords in the peritumoral tissue",
        ),
        PrimitiveSpec(
            "infiltrative-nest-cord-extension-v1",
            "invasion_pattern",
            "increase",
            spatial_scope="boundary",
            morphology="nest_cord",
            zh="沿肿瘤边界增加浸润性巢索延伸",
            en="Increase infiltrative nest-cord extension along the tumor boundary",
        ),
        PrimitiveSpec(
            "invasive-tumor-footprint-decrease-v1",
            "tumor_extent",
            "decrease",
            spatial_scope="whole_lesion",
            zh="缩小整体侵袭性肿瘤面积",
            en="Decrease the invasive tumor footprint across the whole lesion",
        ),
        PrimitiveSpec(
            "local-invasive-clearance-v1",
            "tumor_extent",
            "clear",
            spatial_scope="local",
            zh="清除一个局部侵袭性肿瘤小灶",
            en="Clear a local focus of invasive tumor",
        ),
        PrimitiveSpec(
            "necrosis-appearance-v1",
            "necrosis",
            "appear",
            spatial_scope="intratumoral",
            zh="在肿瘤内部增加坏死",
            en="Increase necrosis within the tumor",
        ),
        PrimitiveSpec(
            "necrosis-resolution-v1",
            "necrosis",
            "repopulate",
            spatial_scope="intratumoral",
            zh="让肿瘤内部的坏死区恢复为存活肿瘤",
            en="Repopulate intratumoral necrosis with viable tumor",
        ),
        PrimitiveSpec(
            "neoplastic-cell-abundance-increase-v1",
            "neoplastic_cell_population",
            "increase",
            spatial_scope="local",
            cell_class="neoplastic",
            zh="增加局部肿瘤细胞数量",
            en="Increase neoplastic cells locally",
        ),
        PrimitiveSpec(
            "neoplastic-cell-abundance-decrease-v1",
            "neoplastic_cell_population",
            "decrease",
            spatial_scope="local",
            cell_class="neoplastic",
            zh="减少局部肿瘤细胞数量",
            en="Decrease neoplastic cells locally",
        ),
        PrimitiveSpec(
            "peritumoral-neoplastic-scatter-increase-v1",
            "invasion_pattern",
            "increase",
            spatial_scope="peritumoral",
            morphology="single_cell",
            zh="在瘤周增加散落的单个肿瘤细胞",
            en="Increase scattered single tumor cells in the peritumoral tissue",
        ),
        PrimitiveSpec(
            "peritumoral-small-cluster-increase-v1",
            "invasion_pattern",
            "increase",
            spatial_scope="peritumoral",
            morphology="small_cluster",
            zh="在瘤周增加肿瘤小细胞簇",
            en="Increase small tumor-cell clusters in the peritumoral tissue",
        ),
        PrimitiveSpec(
            "peritumoral-tumor-nest-formation-v1",
            "invasion_pattern",
            "increase",
            spatial_scope="peritumoral",
            morphology="nest",
            zh="在瘤周形成离散肿瘤巢",
            en="Form discrete tumor nests in the peritumoral tissue",
        ),
        PrimitiveSpec(
            "residual-tumor-fragmentation-v1",
            "tumor_topology",
            "fragment",
            spatial_scope="local",
            morphology="fragmented",
            zh="将局部肿瘤分裂成多个小灶",
            en="Fragment the local tumor into multiple small foci",
        ),
        PrimitiveSpec(
            "stroma-increase-v1",
            "stroma",
            "increase",
            spatial_scope="local",
            zh="在局部增加肿瘤相关间质",
            en="Increase tumor-associated stroma locally",
        ),
    )
}


MULTI_PAIRS = {
    "BCSS": (
        "cohesive-boundary-expansion-v1",
        "neoplastic-cell-abundance-decrease-v1",
    ),
    "PANDA": (
        "infiltrative-nest-cord-extension-v1",
        "peritumoral-neoplastic-scatter-increase-v1",
    ),
    "GLaS": (
        "cellularity-increase-v1",
        "peritumoral-small-cluster-increase-v1",
    ),
    "IGNITE": (
        "necrosis-appearance-v1",
        "generic-immune-infiltrate-increase-v1",
    ),
    "ORCA": (
        "cellularity-decrease-v1",
        "neoplastic-cell-abundance-increase-v1",
    ),
    "PUMA": (
        "cohesive-boundary-expansion-v1",
        "peritumoral-neoplastic-scatter-increase-v1",
    ),
}


def _case_stub(profile: ProfileSpec, instruction: str) -> JointCaseContext:
    return JointCaseContext(
        case_id=f"benchmark-{profile.dataset.casefold()}",
        instruction=instruction,
        source_image_uri="benchmark-image.png",
        source_tissue_mask_uri="benchmark-tissue.npy",
        source_nuclei_mask_uri="benchmark-nuclei.png",
        pathology_domain_id=profile.pathology_domain_id,
        annotation_profile_id=profile.annotation_profile_id,
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id=profile.cell_population_profile_id,
        primitive_id="cohesive-boundary-expansion-v1",
        joint_area_budget=JointAreaBudget(
            target_fraction=0.04,
            min_fraction=0.01,
            max_fraction=0.08,
            tissue_min_fraction=0.01,
        ),
        seed=17,
        provenance={
            "source_image_sha256": "benchmark-image",
            "source_tissue_mask_sha256": "benchmark-tissue",
            "source_nuclei_mask_sha256": "benchmark-nuclei",
        },
    )


def _intent(
    spec: PrimitiveSpec,
    *,
    index: int,
    source_text: str,
    strength: str = "unspecified",
    polarity: str = "affirmed",
) -> SemanticIntentClause:
    return SemanticIntentClause(
        intent_id=f"intent-{index:03d}",
        intent_type=(
            "clinical_trajectory" if spec.clinical_context != "none" else "direct_edit"
        ),
        target=spec.target,
        operation=spec.operation,
        polarity=polarity,
        clinical_context=spec.clinical_context,
        spatial_scope=spec.spatial_scope,
        morphology=spec.morphology,
        cell_class=spec.cell_class,
        strength=strength,
        source_text=source_text,
    )


def _request(
    instruction: str,
    intents: Iterable[SemanticIntentClause],
    relations: Iterable[IntentRelation] = (),
) -> SemanticRequest:
    return SemanticRequest(
        instruction=instruction,
        intents=tuple(intents),
        relations=tuple(relations),
        parser="benchmark_gold_v1",
        parser_metadata={"construction": "catalog-derived bilingual template"},
    )


def _expected_program(request: SemanticRequest, profile: ProfileSpec) -> dict[str, Any]:
    program = SemanticProgramPlanner().plan(
        request,
        case_template=_case_stub(profile, request.instruction),
    )
    return {
        "status": program.status,
        "conflicts": list(program.conflicts),
        "steps": [
            {
                "intent_id": step.intent_id,
                "order_index": step.order_index,
                "depends_on": list(step.depends_on),
                "status": step.status,
                "selected_primitive_id": step.selected_primitive_id,
                "candidates": [
                    {
                        "primitive_id": candidate.primitive_id,
                        "semantic_priority": candidate.semantic_priority,
                        "compatible_mechanism_ids": list(
                            candidate.compatible_mechanism_ids
                        ),
                    }
                    for candidate in step.candidates
                ],
            }
            for step in program.steps
        ],
    }


def _profile_metadata(profile: ProfileSpec) -> dict[str, str]:
    return {
        "dataset": profile.dataset,
        "organ": profile.organ,
        "pathology_domain_id": profile.pathology_domain_id,
        "annotation_profile_id": profile.annotation_profile_id,
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": profile.cell_population_profile_id,
    }


def _record(
    *,
    case_id: str,
    category: str,
    language: str,
    profile: ProfileSpec,
    request: SemanticRequest,
    catalog_target_primitive_ids: Iterable[str],
) -> dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "case_id": case_id,
        "category": category,
        "language": language,
        "instruction": request.instruction,
        "case_profile": _profile_metadata(profile),
        "catalog_target_primitive_ids": list(catalog_target_primitive_ids),
        "gold_semantic_request": request.to_metadata(),
        "expected_planner": _expected_program(request, profile),
    }


def _open_bindings(
    repository: JointSkillRepository,
) -> dict[str, tuple[str, ...]]:
    rows = repository.capability_matrix()
    result: dict[str, tuple[str, ...]] = {}
    for profile in PROFILES:
        primitive_ids: set[str] = set()
        for row in rows:
            if (
                row["pathology_domain_id"] != profile.pathology_domain_id
                or row["annotation_profile_id"] != profile.annotation_profile_id
                or row["status"] in {"unsupported", "render_only"}
            ):
                continue
            mechanism_id = str(row["mechanism_id"])
            for primitive_id in row["supported_primitives"]:
                if (
                    primitive_id in repository.executable_primitive_ids
                    and repository.execution_selection_reason(
                        primitive_id=primitive_id,
                        mechanism_id=mechanism_id,
                    )
                    is None
                ):
                    primitive_ids.add(str(primitive_id))
        result[profile.dataset] = tuple(sorted(primitive_ids))
    return result


def build_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repository = JointSkillRepository()
    open_bindings = _open_bindings(repository)
    open_union = set().union(*(set(items) for items in open_bindings.values()))
    missing_specs = open_union - set(PRIMITIVE_SPECS)
    stale_specs = set(PRIMITIVE_SPECS) - open_union
    if missing_specs or stale_specs:
        raise RuntimeError(
            f"primitive specification mismatch: missing={sorted(missing_specs)}, "
            f"stale={sorted(stale_specs)}"
        )

    records: list[dict[str, Any]] = []
    serial = 0

    def add(**kwargs: Any) -> None:
        nonlocal serial
        serial += 1
        kwargs["case_id"] = f"spp-v1-{serial:04d}"
        records.append(_record(**kwargs))

    # Four bilingual forms for every currently open organ--primitive binding.
    for profile in PROFILES:
        for primitive_id in open_bindings[profile.dataset]:
            spec = PRIMITIVE_SPECS[primitive_id]
            variants = (
                ("zh", spec.zh, "unspecified"),
                ("zh", f"轻微地{spec.zh}", "mild"),
                ("en", spec.en, "unspecified"),
                ("en", f"Mildly {spec.en[0].lower() + spec.en[1:]}", "mild"),
            )
            for language, instruction, strength in variants:
                request = _request(
                    instruction,
                    (
                        _intent(
                            spec, index=1, source_text=instruction, strength=strength
                        ),
                    ),
                )
                expected = _expected_program(request, profile)
                selected = expected["steps"][0]["selected_primitive_id"]
                if selected != primitive_id:
                    raise RuntimeError(
                        f"{profile.dataset}/{primitive_id} resolves to {selected}"
                    )
                add(
                    category="catalog_single_intent",
                    language=language,
                    profile=profile,
                    request=request,
                    catalog_target_primitive_ids=(primitive_id,),
                )

    # Four ordered two-intent instructions per profile, including reverse order.
    for profile in PROFILES:
        left_id, right_id = MULTI_PAIRS[profile.dataset]
        if {left_id, right_id} - set(open_bindings[profile.dataset]):
            raise RuntimeError(f"multi-intent pair is not open for {profile.dataset}")
        for language, reverse in (
            ("zh", False),
            ("zh", True),
            ("en", False),
            ("en", True),
        ):
            first_id, second_id = (
                (right_id, left_id) if reverse else (left_id, right_id)
            )
            first = PRIMITIVE_SPECS[first_id]
            second = PRIMITIVE_SPECS[second_id]
            first_text = first.zh if language == "zh" else first.en
            second_text = second.zh if language == "zh" else second.en
            instruction = (
                f"先{first_text}，然后{second_text}"
                if language == "zh"
                else f"First {first_text[0].lower() + first_text[1:]}, then "
                f"{second_text[0].lower() + second_text[1:]}"
            )
            request = _request(
                instruction,
                (
                    _intent(first, index=1, source_text=first_text),
                    _intent(second, index=2, source_text=second_text),
                ),
                (
                    IntentRelation(
                        before_intent_id="intent-001",
                        after_intent_id="intent-002",
                        relation_type="explicit_sequence",
                    ),
                ),
            )
            add(
                category="ordered_multi_intent",
                language=language,
                profile=profile,
                request=request,
                catalog_target_primitive_ids=(first_id, second_id),
            )

    # Underspecified invasion must preserve uncertainty for mask-aware resolution.
    invasion_spec = PrimitiveSpec(
        "benchmark-unspecified-invasion",
        "invasion_pattern",
        "increase",
        spatial_scope="local",
    )
    for profile in PROFILES:
        for language, instruction in (
            ("zh", "增加局部浸润"),
            ("en", "Increase local invasion"),
        ):
            intent = _intent(invasion_spec, index=1, source_text=instruction)
            intent = SemanticIntentClause(
                **{
                    **intent.to_metadata(),
                    "constraints": (),
                    "uncertainties": ("invasion morphology is unspecified",),
                }
            )
            add(
                category="underspecified_intent",
                language=language,
                profile=profile,
                request=_request(instruction, (intent,)),
                catalog_target_primitive_ids=(),
            )

    # Negated changes must never be promoted into an executable edit.
    negated_spec = PRIMITIVE_SPECS["neoplastic-cell-abundance-increase-v1"]
    for profile in PROFILES:
        for language, instruction in (
            ("zh", "不要增加局部肿瘤细胞"),
            ("en", "Do not increase neoplastic cells locally"),
        ):
            add(
                category="negation",
                language=language,
                profile=profile,
                request=_request(
                    instruction,
                    (
                        _intent(
                            negated_spec,
                            index=1,
                            source_text=instruction,
                            polarity="negated",
                        ),
                    ),
                ),
                catalog_target_primitive_ids=(),
            )

    # Opposing unordered goals must produce a semantic conflict.
    increase_extent = PrimitiveSpec(
        "benchmark-generic-increase",
        "tumor_extent",
        "increase",
    )
    decrease_extent = PrimitiveSpec(
        "benchmark-generic-decrease",
        "tumor_extent",
        "decrease",
    )
    for profile in PROFILES:
        for language, instruction, left_text, right_text in (
            (
                "zh",
                "增加肿瘤面积，同时减少肿瘤面积",
                "增加肿瘤面积",
                "减少肿瘤面积",
            ),
            (
                "en",
                "Increase tumor area and decrease tumor area",
                "Increase tumor area",
                "decrease tumor area",
            ),
        ):
            left = _intent(increase_extent, index=1, source_text=left_text)
            left = SemanticIntentClause(
                **{
                    **left.to_metadata(),
                    "constraints": (),
                    "uncertainties": ("tumor growth morphology is unspecified",),
                }
            )
            request = _request(
                instruction,
                (left, _intent(decrease_extent, index=2, source_text=right_text)),
                (
                    IntentRelation(
                        before_intent_id="intent-001",
                        after_intent_id="intent-002",
                        relation_type="unordered",
                    ),
                ),
            )
            add(
                category="unordered_conflict",
                language=language,
                profile=profile,
                request=request,
                catalog_target_primitive_ids=(),
            )

    expected_count = 4 * sum(map(len, open_bindings.values())) + 60
    if len(records) != expected_count:
        raise RuntimeError(f"expected {expected_count} records, built {len(records)}")

    executable = set(repository.executable_primitive_ids)
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "purpose": "closed-ontology bilingual Parser/Planner interface conformance",
        "scientific_claim_boundary": (
            "synthetic templates and paraphrases; not unrestricted clinical language "
            "understanding and not mask-quality evidence"
        ),
        "record_count": len(records),
        "language_counts": dict(
            sorted(Counter(r["language"] for r in records).items())
        ),
        "category_counts": dict(
            sorted(Counter(r["category"] for r in records).items())
        ),
        "profile_counts": dict(
            sorted(Counter(r["case_profile"]["dataset"] for r in records).items())
        ),
        "open_primitive_type_count": len(open_union),
        "open_organ_profile_binding_count": sum(map(len, open_bindings.values())),
        "open_bindings": {key: list(value) for key, value in open_bindings.items()},
        "executable_scope_without_open_profile_binding": sorted(
            executable - open_union
        ),
        "ontology_schema_version": SEMANTIC_REQUEST_SCHEMA_VERSION,
    }
    return records, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, manifest = build_records()
    jsonl_path = args.output_dir / "benchmark.jsonl"
    body = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    jsonl_path.write_text(body, encoding="utf-8")
    manifest["benchmark_jsonl_sha256"] = hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"jsonl": str(jsonl_path), "manifest": str(manifest_path), **manifest},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

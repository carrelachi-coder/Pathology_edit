"""Read-only qualification of legacy G2 image--instruction pairs.

The qualifier never generates or mutates a target mask.  It freezes source
asset evidence, translates the legacy requested edit into the joint semantic
space, enumerates domain/profile-compatible mechanism candidates and creates
paginated source-only review boards for a later H&E decision.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.visualization import id_mask_to_rgb

from .g2_pilot import ORGAN_CONTRACTS
from .nuclei import iter_instances, load_nuclei_mask, touches_border
from .skills.repository import JointSkillRepository
from .visualization import NUCLEI_RGB, _blend

QUALIFICATION_SCHEMA_VERSION = "g2-read-only-pair-qualification-v1"

# The legacy product manifest mixes tissue-label edits with cellular edits.
# An exact value means that the old instruction already names an executable
# joint primitive.  ``None`` means H&E review must replace the primitive or
# abstain; no silent semantic conversion is allowed.
LEGACY_PRIMITIVE_MAP = {
    "tumor_burden_increase": "tumor-burden-increase-v1",
    "tumor_burden_decrease": "tumor-burden-decrease-v1",
    "stromal_desmoplasia": "stroma-increase-v1",
    "necrosis_appearance": "necrosis-appearance-v1",
    "necrosis_resolution": "necrosis-resolution-v1",
    "stromal_immune_infiltration": "cell-type-abundance-increase-v1",
    "immune_infiltration_decrease": "cell-type-abundance-decrease-v1",
    "stroma_decrease": None,
}

LEGACY_CONVERSION_CLASS = {
    "stromal_immune_infiltration": "immune",
    "immune_infiltration_decrease": "immune",
}

LEGACY_SEMANTICS = {
    "tumor_burden_increase": ("tumor", "increase", "tissue_burden"),
    "tumor_burden_decrease": ("tumor", "decrease", "tissue_burden"),
    "stromal_desmoplasia": ("stroma", "increase", "tissue_compartment"),
    "stroma_decrease": ("stroma", "decrease", "tissue_compartment"),
    "necrosis_appearance": ("necrosis", "increase", "tissue_compartment"),
    "necrosis_resolution": ("necrosis", "decrease", "tissue_compartment"),
    "stromal_immune_infiltration": ("immune", "increase", "cell_population"),
    "immune_infiltration_decrease": ("immune", "decrease", "cell_population"),
}

STROMA_DECREASE_REPLACEMENTS = (
    "tumor-burden-increase-v1",
    "cellularity-increase-v1",
    "neoplastic-cell-infiltration-increase-v1",
)


def qualify_g2_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    board_page_size: int = 12,
    tile_size: int = 176,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("G2 manifest must contain a non-empty cases list")
    repository = JointSkillRepository()
    records = [
        _qualify_row(index, row, repository=repository)
        for index, row in enumerate(rows)
    ]
    board_paths = _write_review_boards(
        records,
        output_dir=output / "source_review_boards",
        page_size=board_page_size,
        tile_size=tile_size,
    )
    by_id = {record["case_id"]: record for record in records}
    for board in board_paths:
        for position, case_id in enumerate(board["case_ids"]):
            by_id[case_id]["review_board"] = {
                "path": board["path"],
                "position": position,
                "page": board["page"],
            }
    jsonl = output / "pair_qualification.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary = _qualification_summary(
        records,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        board_paths=board_paths,
    )
    summary_path = output / "pair_qualification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "records": len(records),
        "jsonl": str(jsonl),
        "summary": str(summary_path),
        "boards": len(board_paths),
        "output_digest": _sha256(jsonl),
    }


def _qualify_row(
    index: int,
    row: dict[str, Any],
    *,
    repository: JointSkillRepository,
) -> dict[str, Any]:
    required = (
        "case_id",
        "instruction",
        "organ",
        "dataset",
        "profile",
        "primitive",
        "source_image",
        "source_tissue_mask",
        "source_nuclei_mask",
    )
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(
            f"G2 row {index} is missing required fields: {', '.join(missing)}"
        )
    organ = str(row["organ"])
    if organ not in ORGAN_CONTRACTS:
        raise ValueError(f"G2 row {index} has unsupported organ {organ!r}")
    domain_id, annotation_id, population_id = ORGAN_CONTRACTS[organ]
    image_path = Path(str(row["source_image"]))
    tissue_path = Path(str(row["source_tissue_mask"]))
    nuclei_path = Path(str(row["source_nuclei_mask"]))
    for path in (image_path, tissue_path, nuclei_path):
        if not path.is_file():
            raise FileNotFoundError(f"source asset does not exist: {path}")
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    tissue = load_id_mask(tissue_path)
    nuclei = load_nuclei_mask(nuclei_path)
    if image.shape[:2] != tissue.shape or tissue.shape != nuclei.shape:
        raise ValueError(f"source assets are not aligned for {row['case_id']}")
    schema = MaskProfileSchema.from_reference_profile(str(row["profile"]))
    tissue_stats = _tissue_statistics(tissue, schema)
    nuclei_stats = _nuclei_statistics(nuclei)
    legacy_primitive = str(row["primitive"])
    if legacy_primitive not in LEGACY_PRIMITIVE_MAP:
        raise ValueError(f"unknown legacy G2 primitive: {legacy_primitive}")
    mapped = LEGACY_PRIMITIVE_MAP[legacy_primitive]
    conversion = legacy_primitive in LEGACY_CONVERSION_CLASS
    if mapped is None:
        status = "requires_primitive_review"
        proposed = list(STROMA_DECREASE_REPLACEMENTS)
        reason = (
            "joint v1 has no stroma-decrease primitive; H&E must establish a "
            "biologically named replacement or abstain"
        )
    elif conversion:
        status = "requires_cell_only_review"
        proposed = [mapped]
        reason = (
            "legacy immune tissue-label edit must be qualified as a complete-instance "
            "cell-population edit before joint execution"
        )
    else:
        status = "ready_for_h&e_review"
        proposed = [mapped]
        reason = "legacy semantic request has an exact joint primitive"
    candidates = {
        primitive_id: _mechanism_candidates(
            repository,
            domain_id=domain_id,
            annotation_id=annotation_id,
            primitive_id=primitive_id,
        )
        for primitive_id in proposed
    }
    if not any(candidates.values()):
        status = "no_registered_mechanism"
        reason = "no domain/profile mechanism candidate is registered"
    expected_mask_digest = row.get("source_mask_sha256")
    actual_tissue_digest = _sha256(tissue_path)
    digest_match = (
        expected_mask_digest is None
        or str(expected_mask_digest) == actual_tissue_digest
    )
    if not digest_match:
        status = "source_digest_mismatch"
        reason = "source tissue digest differs from the legacy manifest"
    subject, direction, scope = LEGACY_SEMANTICS[legacy_primitive]
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "source_index": index,
        "case_id": str(row["case_id"]),
        "sample_id": str(row.get("sample_id") or ""),
        "organ": organ,
        "dataset": str(row["dataset"]),
        "annotation_profile_id": annotation_id,
        "pathology_domain_id": domain_id,
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": population_id,
        "instruction": str(row["instruction"]),
        "legacy_primitive": legacy_primitive,
        "g2_primitive": str(row.get("g2_primitive") or ""),
        "requested_semantics": {
            "subject": subject,
            "direction": direction,
            "scope": scope,
            "explicit_cell_class": LEGACY_CONVERSION_CLASS.get(legacy_primitive),
        },
        "mapped_joint_primitive": mapped,
        "proposed_joint_primitives": proposed,
        "candidate_mechanisms": candidates,
        "qualification_status": status,
        "qualification_reason": reason,
        "requires_h&e_review": True,
        "source_assets": {
            "image": str(image_path),
            "tissue_mask": str(tissue_path),
            "nuclei_mask": str(nuclei_path),
            "shape_hw": list(tissue.shape),
            "aligned": True,
            "image_sha256": _sha256(image_path),
            "tissue_mask_sha256": actual_tissue_digest,
            "nuclei_mask_sha256": _sha256(nuclei_path),
            "legacy_tissue_digest_match": digest_match,
        },
        "source_statistics": {
            "tissue": tissue_stats,
            "nuclei": nuclei_stats,
        },
    }


def _mechanism_candidates(
    repository: JointSkillRepository,
    *,
    domain_id: str,
    annotation_id: str,
    primitive_id: str,
) -> list[dict[str, Any]]:
    profile = repository.annotation_profiles[annotation_id]
    result = []
    for mechanism in repository.mechanisms_for(
        pathology_domain_id=domain_id,
        primitive_id=primitive_id,
    ):
        if mechanism.mechanism_id in profile.unavailable_mechanisms:
            continue
        result.append(
            {
                "mechanism_id": mechanism.mechanism_id,
                "summary": mechanism.summary,
                "required_observations": list(
                    mechanism.recognition.required_observations
                ),
                "contraindications": list(
                    mechanism.recognition.contraindications
                ),
                "profile_support": (
                    "conditional"
                    if mechanism.mechanism_id in profile.conditional_mechanisms
                    else "supported"
                ),
            }
        )
    return result


def _tissue_statistics(
    tissue: np.ndarray, schema: MaskProfileSchema
) -> dict[str, Any]:
    total = int(tissue.size)
    counts = Counter(int(value) for value in tissue.ravel())
    label_counts: dict[str, int] = {}
    known_ids = set(schema.skip_fine_ids)
    for label, ids in schema.label_to_fine_ids.items():
        label_counts[label] = sum(counts.get(int(value), 0) for value in ids)
        known_ids.update(int(value) for value in ids)
    unknown = sorted(set(counts) - known_ids)
    return {
        "fine_id_counts": {str(key): value for key, value in sorted(counts.items())},
        "canonical_label_fractions": {
            key: value / total for key, value in sorted(label_counts.items())
        },
        "unknown_fine_ids": unknown,
        "skip_fraction": sum(counts.get(value, 0) for value in schema.skip_fine_ids)
        / total,
    }


def _nuclei_statistics(nuclei: np.ndarray) -> dict[str, Any]:
    instances = list(iter_instances(nuclei))
    by_class = Counter(class_id for _id, class_id, _mask in instances)
    border = sum(touches_border(mask) for _id, _class, mask in instances)
    return {
        "semantic_pixel_counts": {
            str(class_id): int(np.count_nonzero(nuclei == class_id))
            for class_id in range(1, 6)
        },
        "semantic_fallback_instance_count": len(instances),
        "instance_counts_by_class": {
            str(class_id): by_class.get(class_id, 0)
            for class_id in range(1, 6)
        },
        "border_censored_instance_count": border,
        "instance_authority": "semantic_watershed_fallback",
    }


def _write_review_boards(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    page_size: int,
    tile_size: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["organ"]].append(record)
    boards = []
    per_row = 3
    case_width = tile_size * 3
    case_height = tile_size + 48
    for organ, values in sorted(grouped.items()):
        values.sort(key=lambda item: (item["legacy_primitive"], item["case_id"]))
        for page_index, start in enumerate(range(0, len(values), page_size), start=1):
            page = values[start : start + page_size]
            rows = (len(page) + per_row - 1) // per_row
            canvas = Image.new(
                "RGB", (case_width * per_row, case_height * rows), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for position, record in enumerate(page):
                y = (position // per_row) * case_height
                x = (position % per_row) * case_width
                image = np.asarray(
                    Image.open(record["source_assets"]["image"]).convert("RGB"),
                    dtype=np.uint8,
                )
                tissue = load_id_mask(record["source_assets"]["tissue_mask"])
                nuclei = load_nuclei_mask(record["source_assets"]["nuclei_mask"])
                tissue_view = _blend(image, id_mask_to_rgb(tissue), 0.42)
                nuclei_view = np.array(image, copy=True)
                for class_id, color in NUCLEI_RGB.items():
                    nuclei_view[nuclei == int(class_id)] = np.asarray(
                        color, dtype=np.uint8
                    )
                caption = (
                    f"{record['case_id']} | {record['legacy_primitive']} | "
                    f"{record['qualification_status']}"
                )
                draw.text((x + 4, y + 4), caption, fill="black")
                draw.text((x + 4, y + 24), "H&E | tissue overlay | nuclei overlay", fill="black")
                for column, panel in enumerate((image, tissue_view, nuclei_view)):
                    resized = Image.fromarray(panel).resize(
                        (tile_size, tile_size), Image.Resampling.BILINEAR
                    )
                    canvas.paste(resized, (x + column * tile_size, y + 48))
            path = output_dir / f"{organ}_source_review_{page_index:02d}.jpg"
            canvas.save(path, quality=92, subsampling=0)
            boards.append(
                {
                    "organ": organ,
                    "page": page_index,
                    "path": str(path),
                    "case_ids": [record["case_id"] for record in page],
                }
            )
    index_path = output_dir / "review_board_index.json"
    index_path.write_text(
        json.dumps(boards, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return boards


def _qualification_summary(
    records: list[dict[str, Any]],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    board_paths: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "read_only_contract": {
            "source_assets_opened_read_only": True,
            "target_masks_generated": False,
            "source_manifests_modified": False,
        },
        "case_count": len(records),
        "organ_counts": dict(sorted(Counter(x["organ"] for x in records).items())),
        "legacy_primitive_counts": dict(
            sorted(Counter(x["legacy_primitive"] for x in records).items())
        ),
        "qualification_status_counts": dict(
            sorted(Counter(x["qualification_status"] for x in records).items())
        ),
        "digest_mismatch_count": sum(
            not x["source_assets"]["legacy_tissue_digest_match"] for x in records
        ),
        "unaligned_count": sum(not x["source_assets"]["aligned"] for x in records),
        "unknown_tissue_id_case_count": sum(
            bool(x["source_statistics"]["tissue"]["unknown_fine_ids"])
            for x in records
        ),
        "review_board_count": len(board_paths),
        "h&e_decision_complete": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

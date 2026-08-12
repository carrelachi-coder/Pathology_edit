"""Freeze six materialized annotation datasets into digest-bound evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from phase3_mask_edit_refine.evidence import sha256_file
from phase3_mask_edit_refine.models import RefineContractError

EVIDENCE_FREEZE_SCHEMA = "joint-local-dataset-evidence-index-v1"
MATERIALIZED_PREPROCESSING_SCHEMA = "materialized-patch-preprocessing-v1"

DATASET_PROFILES = {
    "bcss": "bcss-semantic-v1",
    "glas": "glas-gland-v1",
    "ignite": "ignite-semantic-v1",
    "orca": "orca-semantic-v1",
    "panda": "panda-gleason-v1",
    "puma": "puma-semantic-v1",
}

OFFICIAL_SOURCE_IDS = {
    dataset_id: f"dataset-{dataset_id}-official-v1"
    for dataset_id in DATASET_PROFILES
}


def freeze_dataset_evidence(
    grouped_manifest_path: str | Path,
    *,
    output_root: str | Path,
    code_revision: str,
    workers: int = 8,
) -> dict[str, Any]:
    """Hash every materialized image/mask/nucleus triplet and freeze splits."""

    source = Path(grouped_manifest_path)
    payload = _load_json_object(source)
    records_by_dataset = _group_records(payload)
    missing = sorted(set(DATASET_PROFILES) - set(records_by_dataset))
    if missing:
        raise RefineContractError(
            "grouped dataset manifest is missing datasets: " + ", ".join(missing)
        )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    governance = _load_governance()
    source_manifest_sha256 = sha256_file(source)
    frozen = []
    for dataset_id, profile_id in DATASET_PROFILES.items():
        raw_records = records_by_dataset[dataset_id]
        resolved = _resolve_records(raw_records)
        hashed = _hash_records(resolved, workers=workers)
        materialization_evidence = _dataset_materialization_evidence(
            resolved,
            dataset_id=dataset_id,
        )
        preprocessing = _preprocessing_contract(
            payload,
            dataset_id=dataset_id,
            code_revision=code_revision,
            source_manifest_sha256=source_manifest_sha256,
            materialization_evidence=materialization_evidence,
        )
        records = [
            _evidence_record(item, preprocessing=preprocessing)
            for item in hashed
        ]
        revision_core = {
            "dataset_id": dataset_id,
            "annotation_profile_id": profile_id,
            "preprocessing": preprocessing,
            "records": [
                {
                    "record_id": item["record_id"],
                    "split": item["split"],
                    "image_sha256": item["image_sha256"],
                    "mask_sha256": item["mask_sha256"],
                    "nuclei_mask_sha256": item["provenance"][
                        "nuclei_mask_sha256"
                    ],
                }
                for item in records
            ],
        }
        dataset_revision = _sha256_json(revision_core)
        official = governance["sources"][OFFICIAL_SOURCE_IDS[dataset_id]]
        manifest = {
            "schema_version": "joint-local-dataset-evidence-v1",
            "dataset_id": dataset_id,
            "annotation_profile_id": profile_id,
            "dataset_revision": dataset_revision,
            "protocol_sources": [official["uri"]],
            "official_source_id": OFFICIAL_SOURCE_IDS[dataset_id],
            "source_grouped_manifest": str(source.resolve()),
            "source_grouped_manifest_sha256": source_manifest_sha256,
            "preprocessing": preprocessing,
            "materialization_evidence": materialization_evidence,
            "record_count": len(records),
            "split_counts": _split_counts(records),
            "records": records,
        }
        manifest_path = root / f"{profile_id}.evidence.json"
        _write_json(manifest_path, manifest)
        manifest_sha256 = sha256_file(manifest_path)
        manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
            manifest_sha256 + "\n", encoding="utf-8"
        )
        frozen.append(
            {
                "dataset_id": dataset_id,
                "annotation_profile_id": profile_id,
                "dataset_revision": dataset_revision,
                "preprocessing_revision": preprocessing["revision"],
                "record_count": len(records),
                "split_counts": manifest["split_counts"],
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest_sha256,
            }
        )
    index = {
        "schema_version": EVIDENCE_FREEZE_SCHEMA,
        "code_revision": code_revision,
        "source_grouped_manifest": str(source.resolve()),
        "source_grouped_manifest_sha256": source_manifest_sha256,
        "datasets": frozen,
    }
    index_path = root / "evidence_index.json"
    _write_json(index_path, index)
    index_sha256 = sha256_file(index_path)
    index_path.with_suffix(index_path.suffix + ".sha256").write_text(
        index_sha256 + "\n", encoding="utf-8"
    )
    return {
        **index,
        "index_path": str(index_path.resolve()),
        "index_sha256": index_sha256,
    }


def verify_frozen_evidence_index(path: str | Path) -> dict[str, Any]:
    index_path = Path(path)
    index = _load_json_object(index_path)
    if index.get("schema_version") != EVIDENCE_FREEZE_SCHEMA:
        raise RefineContractError("unsupported frozen evidence index schema")
    failures = []
    verified_records = 0
    for dataset in index.get("datasets", []):
        manifest_path = Path(str(dataset.get("manifest_path") or ""))
        if not manifest_path.is_file():
            failures.append(
                {"dataset_id": dataset.get("dataset_id"), "error": "manifest_missing"}
            )
            continue
        if sha256_file(manifest_path) != dataset.get("manifest_sha256"):
            failures.append(
                {"dataset_id": dataset.get("dataset_id"), "error": "manifest_digest"}
            )
            continue
        manifest = _load_json_object(manifest_path)
        materialization = manifest.get("materialization_evidence") or {}
        for artifact_name in ("metadata_jsonl", "stats_txt"):
            artifact = materialization.get(artifact_name) or {}
            target = Path(str(artifact.get("uri") or ""))
            if (
                not target.is_file()
                or sha256_file(target) != artifact.get("sha256")
            ):
                failures.append(
                    {
                        "dataset_id": dataset.get("dataset_id"),
                        "error": f"{artifact_name}_digest",
                    }
                )
        for record in manifest.get("records", []):
            for path_key, digest_key in (
                ("image_uri", "image_sha256"),
                ("mask_uri", "mask_sha256"),
            ):
                target = Path(record[path_key])
                if not target.is_file() or sha256_file(target) != record[digest_key]:
                    failures.append(
                        {
                            "dataset_id": dataset.get("dataset_id"),
                            "record_id": record.get("record_id"),
                            "error": f"{path_key}_digest",
                        }
                    )
            nuclei_path = Path(record["provenance"]["nuclei_mask_uri"])
            if (
                not nuclei_path.is_file()
                or sha256_file(nuclei_path)
                != record["provenance"]["nuclei_mask_sha256"]
            ):
                failures.append(
                    {
                        "dataset_id": dataset.get("dataset_id"),
                        "record_id": record.get("record_id"),
                        "error": "nuclei_mask_digest",
                    }
                )
            verified_records += 1
    return {
        "passed": not failures,
        "verified_records": verified_records,
        "failures": failures,
        "index_sha256": sha256_file(index_path),
    }


def _group_records(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_split, evidence_split in (
        ("train", "train"),
        ("val", "validation"),
        ("test", "test"),
    ):
        records = payload.get(source_split)
        if not isinstance(records, list):
            raise RefineContractError(
                f"grouped dataset manifest lacks {source_split}[]"
            )
        for raw in records:
            if not isinstance(raw, dict):
                raise RefineContractError("grouped dataset record is not an object")
            dataset_id = str(raw.get("dataset_id") or "").lower()
            if dataset_id not in DATASET_PROFILES:
                raise RefineContractError(
                    f"unsupported grouped dataset ID: {dataset_id!r}"
                )
            grouped[dataset_id].append({**raw, "evidence_split": evidence_split})
    return grouped


def _resolve_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = []
    seen = set()
    for raw in sorted(records, key=lambda item: str(item["sample_id"])):
        record_id = str(raw.get("sample_id") or "")
        if not record_id or record_id in seen:
            raise RefineContractError(f"duplicate/empty dataset record ID: {record_id!r}")
        seen.add(record_id)
        dataset_root = Path(str(raw["dataset_root"]))
        image_path = dataset_root / str(raw["images_dir"]) / str(raw["image"])
        mask_path = dataset_root / str(raw["masks_dir"]) / str(raw["mask"])
        nuclei_path = dataset_root / str(raw.get("nuclei_dir", "nuclei_masks")) / str(
            raw["nuclei"]
        )
        missing = [
            str(path)
            for path in (image_path, mask_path, nuclei_path)
            if not path.is_file()
        ]
        if missing:
            raise RefineContractError(
                f"dataset record {record_id} has missing files: {missing}"
            )
        resolved.append(
            {
                **raw,
                "record_id": record_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "nuclei_path": nuclei_path,
            }
        )
    return resolved


def _hash_records(
    records: list[dict[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    paths = sorted(
        {
            Path(item[key])
            for item in records
            for key in ("image_path", "mask_path", "nuclei_path")
        },
        key=str,
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        digests = dict(zip(paths, pool.map(sha256_file, paths), strict=True))
    return [
        {
            **item,
            "image_sha256": digests[item["image_path"]],
            "mask_sha256": digests[item["mask_path"]],
            "nuclei_sha256": digests[item["nuclei_path"]],
        }
        for item in records
    ]


def _preprocessing_contract(
    grouped_payload: dict[str, Any],
    *,
    dataset_id: str,
    code_revision: str,
    source_manifest_sha256: str,
    materialization_evidence: dict[str, Any],
) -> dict[str, Any]:
    upstream = grouped_payload.get("preprocessing_revision")
    evidence_revision = _sha256_json(materialization_evidence)
    core = {
        "schema_version": MATERIALIZED_PREPROCESSING_SCHEMA,
        "dataset_id": dataset_id,
        "code_revision": code_revision,
        "source_manifest_sha256": source_manifest_sha256,
        "split_strategy": grouped_payload.get("strategy"),
        "split_seed": grouped_payload.get("seed"),
        "grouping_authority": "scripts/build_segmentator_multidataset_manifest.py:_group_id",
        "materialized_subdirectories": ["images", "tissue_masks", "nuclei_masks"],
        "upstream_preprocessing_revision": (
            str(upstream) if upstream else evidence_revision
        ),
        "upstream_preprocessing_complete": bool(upstream),
        "materialization_evidence_complete": True,
        "materialization_evidence_revision": evidence_revision,
        "preprocessing_statements": materialization_evidence[
            "preprocessing_statements"
        ],
        "known_limitations": (
            []
            if upstream
            else [
                "The source grouped manifest did not name an upstream preprocessing "
                "code revision. This freeze binds the materialized stats, metadata "
                "and every image/mask/nuclei digest, but does not claim that the "
                "original preprocessing code is reconstructable."
            ]
        ),
    }
    return {**core, "revision": _sha256_json(core)}


def _dataset_materialization_evidence(
    records: list[dict[str, Any]], *, dataset_id: str
) -> dict[str, Any]:
    roots = {Path(str(item["dataset_root"])).resolve() for item in records}
    if len(roots) != 1:
        raise RefineContractError(
            f"dataset {dataset_id} resolves to multiple materialized roots"
        )
    root = next(iter(roots))
    metadata_path = root / "metadata.jsonl"
    stats_path = root / "stats.txt"
    missing = [str(path) for path in (metadata_path, stats_path) if not path.is_file()]
    if missing:
        raise RefineContractError(
            f"dataset {dataset_id} lacks materialization evidence: {missing}"
        )
    stats_text = stats_path.read_text(encoding="utf-8")
    statements = _extract_preprocessing_statements(stats_text)
    if not statements:
        raise RefineContractError(
            f"dataset {dataset_id} stats.txt has no auditable preprocessing statements"
        )
    return {
        "schema_version": "materialized-dataset-evidence-v1",
        "dataset_id": dataset_id,
        "dataset_root": str(root),
        "metadata_jsonl": {
            "uri": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        "stats_txt": {
            "uri": str(stats_path),
            "sha256": sha256_file(stats_path),
        },
        "preprocessing_statements": statements,
    }


def _extract_preprocessing_statements(stats_text: str) -> list[str]:
    prefixes = (
        "Source:",
        "Resize:",
        "Magnification:",
        "Mask quantization:",
        "Patch extraction:",
        "Filter:",
        "Unannotated/context padding filter:",
        "Label remap:",
        "Label scheme:",
        "Organ filter:",
    )
    return [
        line.strip()
        for line in stats_text.splitlines()
        if line.strip().startswith(prefixes)
    ]


def _evidence_record(
    item: dict[str, Any], *, preprocessing: dict[str, Any]
) -> dict[str, Any]:
    group_id = str(item.get("group_id") or "")
    if not group_id:
        raise RefineContractError("dataset record lacks a group_id")
    return {
        "record_id": item["record_id"],
        "mask_uri": str(item["mask_path"].resolve()),
        "image_uri": str(item["image_path"].resolve()),
        "patient_id": group_id,
        "wsi_id": group_id,
        "split": item["evidence_split"],
        "mask_sha256": item["mask_sha256"],
        "image_sha256": item["image_sha256"],
        "provenance": {
            "preprocessing_revision": preprocessing["revision"],
            "upstream_preprocessing_revision": preprocessing[
                "upstream_preprocessing_revision"
            ],
            "group_id": group_id,
            "group_authority": preprocessing["grouping_authority"],
            "nuclei_mask_uri": str(item["nuclei_path"].resolve()),
            "nuclei_mask_sha256": item["nuclei_sha256"],
            "original_label_map_digest": item["mask_sha256"],
        },
    }


def _split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    values: dict[str, int] = defaultdict(int)
    for record in records:
        values[str(record["split"])] += 1
    return dict(sorted(values.items()))


def _load_governance() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parent
        / "skills"
        / "catalog"
        / "evidence-governance-v2.json"
    )
    return _load_json_object(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefineContractError(f"could not load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RefineContractError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

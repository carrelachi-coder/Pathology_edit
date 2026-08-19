"""Deterministic manifest builder for the committed mask-skill catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from phase3_mask_edit_refine.models import RefineContractError

CATALOG_MANIFEST_SCHEMA_VERSION = "mask-skill-catalog-manifest-v1"
OFFICIAL_CATALOG_ROOT = Path(__file__).resolve().parent / "catalog"
OFFICIAL_CATALOG_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "catalog_manifest_v1.json"
)


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_catalog_manifest(catalog_root: Path = OFFICIAL_CATALOG_ROOT) -> dict[str, Any]:
    root = catalog_root.resolve()
    packages: list[dict[str, Any]] = []
    for rules_path in sorted(root.glob("*/*/references/rules.json")):
        try:
            rules_payload = json.loads(rules_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RefineContractError(
                f"could not build catalog manifest from {rules_path}: {exc}"
            ) from exc
        if not isinstance(rules_payload, dict):
            raise RefineContractError(
                f"catalog rules root must be an object: {rules_path}"
            )
        contract_path = rules_path.with_name("mask_contract.json")
        if not contract_path.is_file():
            raise RefineContractError(
                f"official skill package lacks mask_contract.json: {rules_path}"
            )
        try:
            contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RefineContractError(
                f"could not build catalog manifest from {contract_path}: {exc}"
            ) from exc
        if not isinstance(contract_payload, dict) or not isinstance(
            contract_payload.get("constraints"), list
        ):
            raise RefineContractError(
                f"catalog mask contract is invalid: {contract_path}"
            )
        if contract_payload.get("skill_id") != rules_payload.get("skill_id"):
            raise RefineContractError(
                f"catalog rules/contract skill mismatch: {rules_path}"
            )
        merged_payload = dict(rules_payload)
        merged_payload["mask_constraints"] = contract_payload["constraints"]
        packages.append(
            {
                "skill_id": rules_payload.get("skill_id"),
                "skill_kind": rules_payload.get("skill_kind"),
                "version": rules_payload.get("version"),
                "rules_path": rules_path.relative_to(root).as_posix(),
                "rules_sha256": file_sha256(rules_path),
                "mask_contract_path": contract_path.relative_to(root).as_posix(),
                "mask_contract_sha256": file_sha256(contract_path),
                "package_digest_sha256": canonical_json_sha256(merged_payload),
            }
        )
    if not packages:
        raise RefineContractError(f"official catalog is empty: {root}")
    skill_ids = [str(item["skill_id"]) for item in packages]
    if len(skill_ids) != len(set(skill_ids)):
        raise RefineContractError("official catalog manifest has duplicate skill IDs")
    catalog_content_sha256 = canonical_json_sha256(packages)
    return {
        "schema_version": CATALOG_MANIFEST_SCHEMA_VERSION,
        "catalog_content_sha256": catalog_content_sha256,
        "package_count": len(packages),
        "packages": packages,
    }


def load_verified_official_catalog_manifest() -> tuple[dict[str, Any], str]:
    path = OFFICIAL_CATALOG_MANIFEST_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefineContractError(
            f"could not load official catalog manifest {path}: {exc}"
        ) from exc
    expected = build_catalog_manifest(OFFICIAL_CATALOG_ROOT)
    if payload != expected:
        raise RefineContractError(
            "official mask-skill catalog is detached from its committed manifest"
        )
    return payload, file_sha256(path)

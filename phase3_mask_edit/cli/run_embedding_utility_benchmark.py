#!/usr/bin/env python3
"""Run a frozen paired Cross/Inpaint embedding-displacement analysis."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from phase3_mask_edit.benchmark.embedding_utility import (
    compute_embedding_utility_scores,
    summarize_scores,
)
from phase3_mask_edit.benchmark.pathokid import (
    build_conch_extractor,
    build_uni2h_extractor,
    input_digest,
    load_feature_cache,
    save_feature_cache,
    sha256_file,
    stable_digest,
)


DEFAULT_CONFIG = Path("benchmark_configs/pathokid.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--feature-extractor", choices=("uni2h", "conch"), default="uni2h"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--wsi-field", default="wsi_id")
    parser.add_argument("--reference-field", default="reference_image")
    parser.add_argument("--inpaint-field", default="inpaint_image")
    parser.add_argument("--cross-field", default="cross_image")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args(argv)


def load_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"unsupported manifest structure: {path}")
    return records


def validate_records(
    records: list[dict],
    *,
    sample_id_field: str,
    wsi_field: str,
    image_fields: tuple[str, str, str],
    expected_count: int | None,
    expected_size: tuple[int, int],
) -> tuple[list[str], list[str], dict[str, list[Path]]]:
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"frozen utility cohort requires {expected_count} rows, found {len(records)}"
        )
    sample_ids: list[str] = []
    groups: list[str] = []
    paths = {field: [] for field in image_fields}
    for index, record in enumerate(records):
        sample_id = str(record.get(sample_id_field) or "")
        group = str(record.get(wsi_field) or "")
        if not sample_id or not group:
            raise ValueError(
                f"row {index}: missing {sample_id_field!r} or {wsi_field!r}"
            )
        sample_ids.append(sample_id)
        groups.append(group)
        for field in image_fields:
            value = record.get(field)
            if value in (None, ""):
                raise ValueError(f"{sample_id}: missing image field {field!r}")
            paths[field].append(Path(str(value)))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample IDs must be unique")
    for field, image_paths in paths.items():
        if len(set(map(str, image_paths))) != len(image_paths):
            raise ValueError(f"{field} paths must be unique")
        _validate_images(image_paths, expected_size=expected_size, field=field)
    return sample_ids, groups, paths


def _validate_images(
    image_paths: list[Path], *, expected_size: tuple[int, int], field: str
) -> None:
    failures = []
    for path in image_paths:
        if not path.is_file():
            failures.append(f"missing: {path}")
        else:
            try:
                with Image.open(path) as image:
                    if image.size != expected_size:
                        failures.append(f"{path}: size={image.size}")
            except Exception as exc:
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
        if len(failures) >= 20:
            break
    if failures:
        raise ValueError(
            f"invalid {field} images; expected RGB-compatible {expected_size} patches:\n"
            + "\n".join(failures)
        )


def get_or_extract_features(
    *,
    extractor,
    extractor_digest: str,
    set_name: str,
    sample_ids: list[str],
    image_paths: list[Path],
    cache_root: Path,
    batch_size: int,
    overwrite: bool,
    extractor_name: str,
) -> tuple[np.ndarray, bool]:
    feature_path = cache_root / f"{set_name}.npz"
    metadata_path = cache_root / f"{set_name}.json"
    digest = input_digest(sample_ids, image_paths)
    if not overwrite:
        cached = load_feature_cache(
            feature_path,
            metadata_path,
            expected_sample_ids=sample_ids,
            expected_input_digest=digest,
            expected_extractor_digest=extractor_digest,
        )
        if cached is not None:
            print(f"[{extractor_name}] cache hit: {set_name} {cached.shape}", flush=True)
            return cached, True

    def progress(done: int, total: int) -> None:
        if done == total or done % max(batch_size, 50) == 0:
            print(f"[{extractor_name}] {set_name}: {done}/{total}", flush=True)

    features = extractor.extract(
        image_paths,
        batch_size=batch_size,
        progress=progress,
    )
    save_feature_cache(
        feature_path,
        metadata_path,
        sample_ids=sample_ids,
        features=features,
        metadata={
            "schema_version": 1,
            "set_name": set_name,
            "sample_count": len(sample_ids),
            "feature_shape": list(features.shape),
            "input_digest": digest,
            "extractor_digest": extractor_digest,
            "extractor": extractor.metadata,
            "features_are_l2_normalized": False,
        },
    )
    return features, False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    frame = config["evaluation_frame"]
    extractor_name = args.feature_extractor
    extractor_cfg = config["feature_extractors"][extractor_name]
    dtype_name = args.dtype or evaluation["dtype"]
    batch_size = args.batch_size or int(evaluation["batch_size"])
    records = load_records(args.manifest)
    if args.max_items is not None:
        records = records[: args.max_items]
    expected_count = None if args.max_items is not None else args.expected_count
    sample_ids, groups, paths = validate_records(
        records,
        sample_id_field=args.sample_id_field,
        wsi_field=args.wsi_field,
        image_fields=(args.reference_field, args.inpaint_field, args.cross_field),
        expected_count=expected_count,
        expected_size=tuple(int(value) for value in frame["image_size"]),
    )
    if len(sample_ids) < 3:
        raise ValueError("embedding utility requires at least three paired samples")

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(extractor_cfg["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint)
    if extractor_name == "conch":
        extractor = build_conch_extractor(
            Path(extractor_cfg["root"]),
            checkpoint,
            device=args.device,
            dtype_name=dtype_name,
            checkpoint_sha256=checkpoint_sha256,
        )
    else:
        extractor = build_uni2h_extractor(
            Path(extractor_cfg["root"]),
            device=args.device,
            dtype_name=dtype_name,
            checkpoint_sha256=checkpoint_sha256,
        )
    extractor_digest = stable_digest(extractor.metadata)
    features: dict[str, np.ndarray] = {}
    cache_hits: dict[str, bool] = {}
    for set_name, field in (
        ("reference", args.reference_field),
        ("inpaint", args.inpaint_field),
        ("cross", args.cross_field),
    ):
        features[set_name], cache_hits[set_name] = get_or_extract_features(
            extractor=extractor,
            extractor_digest=extractor_digest,
            set_name=set_name,
            sample_ids=sample_ids,
            image_paths=paths[field],
            cache_root=args.output_root / "cache" / extractor_name,
            batch_size=batch_size,
            overwrite=args.overwrite_cache,
            extractor_name=extractor_name,
        )

    scores = compute_embedding_utility_scores(
        features["reference"], features["inpaint"], features["cross"]
    )
    metric_vectors = {
        "inpaint_directional_consistency": scores.inpaint_directional_consistency,
        "cross_directional_consistency": scores.cross_directional_consistency,
        "paired_backend_agreement": scores.paired_backend_agreement,
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "analysis": f"exploratory_bcss_moderate_tumor_increase_{extractor_name}",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(sample_ids),
        "wsi_count": len(set(groups)),
        "feature_policy": f"raw_{extractor_name}_difference_then_row_l2_for_cosine",
        "extractor": {**extractor.metadata, "extractor_digest": extractor_digest},
        "cache_hits": cache_hits,
        "bootstrap": {
            "unit": args.wsi_field,
            "repeats": args.bootstrap_repeats,
            "seed": args.seed,
        },
        "metrics": {
            name: summarize_scores(
                values,
                groups,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed + index * 1009,
            )
            for index, (name, values) in enumerate(metric_vectors.items())
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_root / "embedding_utility_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_root / "embedding_utility_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "sample_id",
            "wsi_id",
            "inpaint_directional_consistency",
            "cross_directional_consistency",
            "paired_backend_agreement",
            "inpaint_displacement_norm",
            "cross_displacement_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "wsi_id": groups[index],
                    **{
                        name: float(values[index])
                        for name, values in metric_vectors.items()
                    },
                    "inpaint_displacement_norm": float(
                        scores.inpaint_displacement_norm[index]
                    ),
                    "cross_displacement_norm": float(
                        scores.cross_displacement_norm[index]
                    ),
                }
            )
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

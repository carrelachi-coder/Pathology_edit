#!/usr/bin/env python3
"""Run cached UNI-2h and CONCH Patho-KID evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
from itertools import combinations
import json
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np
from PIL import Image
import yaml

from phase3_mask_edit.benchmark.pathokid import (
    build_conch_extractor,
    build_uni2h_extractor,
    cluster_bootstrap_kid,
    input_digest,
    load_feature_cache,
    make_cluster_bootstrap_draws,
    paired_bootstrap_delta,
    save_feature_cache,
    sha256_file,
    stable_digest,
    subset_kid,
    summarize_values,
    unbiased_kid,
)


DEFAULT_CONFIG = Path("benchmark_configs/pathokid.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        action="append",
        required=True,
        metavar="MODEL_ID=PATH",
        help="Repeat once per model; PATH contains organ/sample_id/generated.png.",
    )
    parser.add_argument("--feature-extractors", nargs="+", choices=("uni2h", "conch"))
    parser.add_argument("--real-path-field", default="target_image")
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--organ-field", default="organ")
    parser.add_argument("--bootstrap-group-field", default="wsi_id")
    parser.add_argument("--generated-filename", default="generated.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--subset-repeats", type=int)
    parser.add_argument("--bootstrap-repeats", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
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


def parse_model_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-root must be MODEL_ID=PATH, got {value!r}")
        model_id, path = value.split("=", 1)
        if not model_id or model_id in roots:
            raise ValueError(f"invalid or duplicate model id: {model_id!r}")
        roots[model_id] = Path(path)
    return roots


def require_record_field(record: dict, field: str, sample_id: str):
    value = record.get(field)
    if value in (None, ""):
        raise ValueError(f"{sample_id}: missing manifest field {field!r}")
    return value


def validate_inputs(
    records: list[dict],
    *,
    model_roots: dict[str, Path],
    sample_id_field: str,
    organ_field: str,
    real_path_field: str,
    bootstrap_group_field: str,
    generated_filename: str,
    expected_image_size: tuple[int, int],
) -> tuple[list[str], list[str], list[str], list[Path], dict[str, list[Path]]]:
    sample_ids = []
    organs = []
    groups = []
    real_paths = []
    model_paths = {model_id: [] for model_id in model_roots}
    for record in records:
        sample_id = str(require_record_field(record, sample_id_field, "unknown"))
        organ = str(require_record_field(record, organ_field, sample_id))
        group = str(require_record_field(record, bootstrap_group_field, sample_id))
        real_path = Path(require_record_field(record, real_path_field, sample_id))
        sample_ids.append(sample_id)
        organs.append(organ)
        groups.append(group)
        real_paths.append(real_path)
        for model_id, root in model_roots.items():
            model_paths[model_id].append(
                root / organ / sample_id / generated_filename
            )

    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample IDs must be unique")
    if len(set(map(str, real_paths))) != len(real_paths):
        raise ValueError("real target paths must be unique")
    missing = [str(path) for path in real_paths if not path.is_file()]
    for model_id, paths in model_paths.items():
        missing.extend(f"{model_id}: {path}" for path in paths if not path.is_file())
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} evaluation images:\n{preview}")
    validate_image_frame(
        real_paths
        + [path for paths in model_paths.values() for path in paths],
        expected_size=expected_image_size,
    )
    return sample_ids, organs, groups, real_paths, model_paths


def validate_image_frame(
    image_paths: list[Path], *, expected_size: tuple[int, int]
) -> None:
    expected_size = (int(expected_size[0]), int(expected_size[1]))
    failures = []
    for path in dict.fromkeys(image_paths):
        try:
            with Image.open(path) as image:
                if image.size != expected_size:
                    failures.append(f"{path}: {image.size}")
        except Exception as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
        if len(failures) >= 20:
            break
    if failures:
        raise ValueError(
            f"Patho-KID requires normalized {expected_size} patches; "
            f"invalid inputs:\n" + "\n".join(failures)
        )


def organ_counts(organs: list[str]) -> dict[str, int]:
    return {
        organ: sum(value == organ for value in organs)
        for organ in sorted(set(organs))
    }


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
) -> tuple[np.ndarray, bool]:
    set_root = cache_root / extractor.name
    feature_path = set_root / f"{set_name}.npz"
    metadata_path = set_root / f"{set_name}.json"
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
            print(f"[{extractor.name}] cache hit: {set_name} {cached.shape}", flush=True)
            return cached, True

    last_reported = -1

    def progress(done: int, total: int) -> None:
        nonlocal last_reported
        if done == total or done - last_reported >= max(batch_size, 50):
            print(f"[{extractor.name}] {set_name}: {done}/{total}", flush=True)
            last_reported = done

    features = extractor.extract(
        image_paths, batch_size=batch_size, progress=progress
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
            "feature_dtype": str(features.dtype),
            "input_digest": digest,
            "extractor_digest": extractor_digest,
            "extractor": extractor.metadata,
            "features_are_l2_normalized": False,
        },
    )
    return features, False


def prefixed_summary(values: np.ndarray) -> dict:
    summary = summarize_values(values)
    return {
        **summary,
        "mean_x1000": summary["mean"] * 1000.0,
        "std_x1000": summary["std"] * 1000.0,
        "ci95_low_x1000": summary["ci95_low"] * 1000.0,
        "ci95_high_x1000": summary["ci95_high"] * 1000.0,
    }


def runtime_provenance(device: str) -> dict:
    import PIL
    import timm
    import torch
    import torchvision

    gpu = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        gpu = torch.cuda.get_device_name(torch.device(device))
    return {
        "command": shlex.join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "device": device,
        "gpu": gpu,
    }


def main() -> int:
    started = time.time()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    evaluation_frame = config["evaluation_frame"]
    expected_image_size = tuple(
        int(value) for value in evaluation_frame["image_size"]
    )
    extractors_config = config["feature_extractors"]
    extractor_names = args.feature_extractors or list(extractors_config)
    dtype_name = args.dtype or evaluation["dtype"]
    batch_size = args.batch_size or int(evaluation["batch_size"])
    subset_repeats = args.subset_repeats or int(evaluation["subset_repeats"])
    bootstrap_repeats = args.bootstrap_repeats or int(
        evaluation["bootstrap_repeats"]
    )
    seed = args.seed if args.seed is not None else int(evaluation["seed"])

    records = load_records(args.manifest)
    if args.max_items is not None:
        records = records[: args.max_items]
    if len(records) < 2:
        raise ValueError("Patho-KID requires at least two manifest rows")
    model_roots = parse_model_roots(args.model_root)
    sample_ids, organs, groups, real_paths, model_paths = validate_inputs(
        records,
        model_roots=model_roots,
        sample_id_field=args.sample_id_field,
        organ_field=args.organ_field,
        real_path_field=args.real_path_field,
        bootstrap_group_field=args.bootstrap_group_field,
        generated_filename=args.generated_filename,
        expected_image_size=expected_image_size,
    )
    subset_size = args.subset_size or min(
        int(evaluation["subset_size"]), len(sample_ids)
    )
    if subset_size < 2:
        raise ValueError("subset size must be at least two")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = sha256_file(args.manifest)
    sample_ids_sha256 = stable_digest(sample_ids)
    draws = make_cluster_bootstrap_draws(
        groups, repeats=bootstrap_repeats, seed=seed
    )
    draws_path = args.output_root / "bootstrap_draws.json"
    draws_path.write_text(
        json.dumps(
            {
                **draws.to_json(),
                "bootstrap_group_field": args.bootstrap_group_field,
                "sample_ids_sha256": sample_ids_sha256,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows_path = args.output_root / "evaluation_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(sample_ids):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "organ": organs[index],
                        "bootstrap_group": groups[index],
                        "real_image": str(real_paths[index]),
                        "generated_images": {
                            model_id: str(paths[index])
                            for model_id, paths in model_paths.items()
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "schema_version": 1,
        "status": "running",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "sample_count": len(sample_ids),
        "sample_ids_sha256": sample_ids_sha256,
        "organ_counts": organ_counts(organs),
        "bootstrap_group_field": args.bootstrap_group_field,
        "bootstrap_group_count": len(draws.group_names),
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": seed,
        "bootstrap_draws": str(draws_path),
        "evaluation_rows": str(rows_path),
        "subset_size": subset_size,
        "subset_repeats": subset_repeats,
        "feature_normalization": "none_raw_activations",
        "kernel": "(x^T y / d + 1)^3",
        "evaluation_frame": evaluation_frame,
        "models": {key: str(path) for key, path in model_roots.items()},
        "feature_extractors": {},
        "results": {},
        "pairwise_model_deltas": {},
        "runtime": runtime_provenance(args.device),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.output_root / "pathokid_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    checkpoint_hashes = {}
    for name in extractor_names:
        cfg = extractors_config[name]
        checkpoint = Path(cfg["checkpoint"])
        print(f"[{name}] hashing checkpoint {checkpoint}", flush=True)
        checkpoint_hashes[name] = sha256_file(checkpoint)

    summary_rows = []
    pairwise_rows = []
    for extractor_index, name in enumerate(extractor_names):
        cfg = extractors_config[name]
        if name == "uni2h":
            extractor = build_uni2h_extractor(
                Path(cfg["root"]),
                device=args.device,
                dtype_name=dtype_name,
                checkpoint_sha256=checkpoint_hashes[name],
            )
        else:
            extractor = build_conch_extractor(
                Path(cfg["root"]),
                Path(cfg["checkpoint"]),
                device=args.device,
                dtype_name=dtype_name,
                checkpoint_sha256=checkpoint_hashes[name],
            )
        extractor_digest = stable_digest(extractor.metadata)
        report["feature_extractors"][name] = {
            **extractor.metadata,
            "extractor_digest": extractor_digest,
        }
        raw_real, _ = get_or_extract_features(
            extractor=extractor,
            extractor_digest=extractor_digest,
            set_name="real",
            sample_ids=sample_ids,
            image_paths=real_paths,
            cache_root=args.output_root / "cache",
            batch_size=batch_size,
            overwrite=args.overwrite_cache,
        )
        real = np.asarray(raw_real, dtype=np.float64)
        extractor_results = {}
        bootstrap_vectors = {}
        for model_index, (model_id, paths) in enumerate(model_paths.items()):
            raw_generated, cache_hit = get_or_extract_features(
                extractor=extractor,
                extractor_digest=extractor_digest,
                set_name=model_id,
                sample_ids=sample_ids,
                image_paths=paths,
                cache_root=args.output_root / "cache",
                batch_size=batch_size,
                overwrite=args.overwrite_cache,
            )
            generated = np.asarray(raw_generated, dtype=np.float64)
            full = unbiased_kid(real, generated)
            subsets = subset_kid(
                real,
                generated,
                subset_size=subset_size,
                repeats=subset_repeats,
                seed=seed + 1009 * extractor_index + 37 * model_index,
            )
            bootstrap = cluster_bootstrap_kid(real, generated, groups, draws)
            bootstrap_vectors[model_id] = bootstrap
            result = {
                "sample_count": len(sample_ids),
                "feature_dimension": int(real.shape[1]),
                "generated_cache_hit": cache_hit,
                "kid_full": full,
                "kid_full_x1000": full * 1000.0,
                "subset": {
                    "size": subset_size,
                    "repeats": subset_repeats,
                    **prefixed_summary(subsets),
                },
                "cluster_bootstrap": prefixed_summary(bootstrap),
            }
            extractor_results[model_id] = result
            summary_rows.append(
                {
                    "extractor": name,
                    "model_id": model_id,
                    "sample_count": len(sample_ids),
                    "feature_dimension": int(real.shape[1]),
                    "kid_full": full,
                    "kid_full_x1000": full * 1000.0,
                    "bootstrap_mean": result["cluster_bootstrap"]["mean"],
                    "bootstrap_ci95_low": result["cluster_bootstrap"]["ci95_low"],
                    "bootstrap_ci95_high": result["cluster_bootstrap"]["ci95_high"],
                }
            )
            print(
                f"[{name}] {model_id}: KID={full:.8g}, "
                f"bootstrap95=[{result['cluster_bootstrap']['ci95_low']:.8g}, "
                f"{result['cluster_bootstrap']['ci95_high']:.8g}]",
                flush=True,
            )
        pairwise_results = {}
        for left_model, right_model in combinations(model_paths, 2):
            delta = paired_bootstrap_delta(
                bootstrap_vectors[left_model], bootstrap_vectors[right_model]
            )
            comparison_id = f"{left_model}__minus__{right_model}"
            pairwise_results[comparison_id] = {
                "left_model": left_model,
                "right_model": right_model,
                "delta_definition": "KID(left) - KID(right); negative favors left",
                **delta,
            }
            pairwise_rows.append(
                {
                    "extractor": name,
                    "left_model": left_model,
                    "right_model": right_model,
                    "delta_mean": delta["mean"],
                    "delta_ci95_low": delta["ci95_low"],
                    "delta_ci95_high": delta["ci95_high"],
                    "probability_left_better": delta["probability_left_better"],
                }
            )
        report["results"][name] = extractor_results
        report["pairwise_model_deltas"][name] = pairwise_results
        del extractor
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary_path = args.output_root / "pathokid_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    pairwise_path = args.output_root / "pathokid_pairwise_deltas.csv"
    if pairwise_rows:
        with pairwise_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
            writer.writeheader()
            writer.writerows(pairwise_rows)
    report["status"] = "completed"
    report["summary_csv"] = str(summary_path)
    report["pairwise_deltas_csv"] = str(pairwise_path)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["runtime_seconds"] = round(time.time() - started, 3)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(report_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

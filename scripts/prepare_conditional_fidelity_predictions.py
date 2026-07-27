#!/usr/bin/env python3
"""Stage generated images and emit reproducible evaluator inference commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

from PIL import Image
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("benchmark_configs/conditional_fidelity.yaml")
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python", default="/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python"
    )
    parser.add_argument(
        "--tissue-python",
        default="/home/lyw/anaconda3/envs/pathology-segmentator-mmseg/bin/python3.10",
    )
    parser.add_argument("--cuda-visible-device", default="4")
    parser.add_argument("--overwrite-links", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"unsupported manifest: {path}")
    return records


def command(parts: list[object], cuda_visible_device: str) -> str:
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_device)} " + shlex.join(
        [str(part) for part in parts]
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    eligible_models = [
        model_id
        for model_id, model_config in config["models"].items()
        if model_config.get("condition_structure_metrics")
    ]
    models = args.models or eligible_models
    unknown = sorted(set(models) - set(config["models"]))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    ineligible = sorted(set(models) - set(eligible_models))
    if ineligible:
        raise ValueError(
            "target-structure predictions require target geometry; "
            f"ineligible models: {ineligible}"
        )
    records = load_records(args.manifest)
    evaluation_size = tuple(int(value) for value in config["evaluation_frame"]["image_size"])
    stage_root = args.output_root / "staged_images"
    manifest_root = args.output_root / "prediction_manifests"
    prediction_root = args.output_root / "predictions"
    commands = []
    staged = 0
    missing = []
    grouped: dict[str, list[dict]] = {}
    for model_id in models:
        allowed_organs = set(config["models"][model_id].get("organs", []))
        model_records = [
            record
            for record in records
            if not allowed_organs or record["organ"] in allowed_organs
        ]
        for record in model_records:
            source = (
                args.generated_root
                / model_id
                / record["organ"]
                / record["sample_id"]
                / "generated.png"
            )
            if not source.is_file():
                missing.append({"model_id": model_id, "sample_id": record["sample_id"], "path": str(source)})
                continue
            with Image.open(source) as image:
                if image.size != evaluation_size:
                    raise ValueError(
                        f"{model_id}/{record['sample_id']}: expected normalized "
                        f"{evaluation_size} patch, got {image.size}: {source}"
                    )
            destination = stage_root / model_id / f"{record['sample_id']}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                if args.overwrite_links:
                    destination.unlink()
                elif destination.resolve() == source.resolve():
                    grouped.setdefault(model_id, []).append(record)
                    continue
                else:
                    raise FileExistsError(destination)
            os.symlink(source.resolve(), destination)
            staged += 1
            grouped.setdefault(model_id, []).append(record)
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} generated images; first entries: {missing[:10]}"
        )

    tissue_checkpoint = config["tissue"]["evaluator"]["checkpoint"]
    cellvit_checkpoint = config["cellvit"]["checkpoint"]
    conic_checkpoint = config["conic"]["checkpoint"]
    for model_id, rows in sorted(grouped.items()):
        input_dir = stage_root / model_id
        tissue_dir = prediction_root / "tissue" / model_id
        cellvit_mask_dir = prediction_root / "cellvit_masks" / model_id
        cellvit_json_dir = prediction_root / "cellvit" / model_id
        cellvit_raw_dir = prediction_root / "cellvit_raw" / model_id
        commands.append(
            {
                "model_id": model_id,
                "organ": "all",
                "evaluator": "tissue_segmentator",
                "command": command(
                    [
                        args.tissue_python,
                        args.repo_root / "scripts/predict_segmentator_masks_batch.py",
                        "--checkpoint",
                        tissue_checkpoint,
                        "--input-dir",
                        input_dir,
                        "--output-dir",
                        tissue_dir,
                        "--batch-size",
                        2,
                        "--decoder",
                        "mask2former",
                        "--device",
                        "cuda:0",
                        "--skip-existing",
                    ],
                    args.cuda_visible_device,
                ),
            }
        )
        commands.append(
            {
                "model_id": model_id,
                "organ": "all",
                "evaluator": "cellvit",
                "command": command(
                    [
                        args.python,
                        args.repo_root / "batch_cellvit_ui_equiv.py",
                        "--images-dir",
                        input_dir,
                        "--output-dir",
                        cellvit_mask_dir,
                        "--json-output-dir",
                        cellvit_json_dir,
                        "--raw-outdir",
                        cellvit_raw_dir,
                        "--model",
                        cellvit_checkpoint,
                        "--gpu",
                        0,
                        "--batch-size",
                        8,
                        "--mpp",
                        0.25,
                        "--magnification",
                        40,
                        "--resolution",
                        0.25,
                        "--skip-existing",
                    ],
                    args.cuda_visible_device,
                ),
            }
        )
        if config["models"][model_id].get("strict_spatial_evaluator") == "conic":
            prediction_manifest = manifest_root / f"{model_id}.json"
            prediction_manifest.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "records": [
                    {
                        "sample_id": row["sample_id"],
                        "generated_image": str(input_dir / f"{row['sample_id']}.png"),
                    }
                    for row in rows
                ]
            }
            prediction_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            commands.append(
                {
                    "model_id": model_id,
                    "organ": "all",
                    "evaluator": "conic_hovernet",
                    "command": command(
                        [
                            args.python,
                            args.repo_root / "scripts/prepare_pathdiff_conic_masks.py",
                            "--manifest",
                            prediction_manifest,
                            "--checkpoint",
                            conic_checkpoint,
                            "--output-root",
                            prediction_root / "conic" / model_id,
                            "--image-field",
                            "generated_image",
                            "--id-field",
                            "sample_id",
                            "--device",
                            "cuda:0",
                            "--batch-size",
                            4,
                        ],
                        args.cuda_visible_device,
                    ),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "generated_root": str(args.generated_root.resolve()),
        "models": models,
        "evaluation_frame": config["evaluation_frame"],
        "record_count": len(records),
        "model_record_counts": {
            model_id: len(rows) for model_id, rows in sorted(grouped.items())
        },
        "staged_links_created": staged,
        "command_count": len(commands),
        "commands": commands,
    }
    output = args.output_root / "prediction_commands.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    shell = args.output_root / "run_predictions.sh"
    shell.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(item["command"] for item in commands)
        + "\n",
        encoding="utf-8",
    )
    shell.chmod(0o755)
    print(json.dumps({"staged": staged, "commands": len(commands), "shell": str(shell)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

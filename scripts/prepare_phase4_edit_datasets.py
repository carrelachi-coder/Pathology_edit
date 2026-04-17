#!/usr/bin/env python3
"""Stage local patch masks into a server-uploadable Phase 4 edit_datasets tree."""

import argparse
import json
import os
import shutil
from pathlib import Path


DATASET_PATCH_DIRS = {
    "BCSS": ("BCSS", "BCSS_PATCHES"),
    "GlaS": ("GLAS", "GlaS_PATCHES"),
    "IGNITE": (None, "IGNITE_PATCHES"),
    "ORCA": ("ORCA", "ORCA_PATCHES"),
    "PANDA": ("PANDA", "PANDA_PATCHES"),
    "PUMA": ("PUMA", "PUMA_PATCHES"),
}


def link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def iter_files(path):
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file())


def stage_dataset(source_root, output_root, dataset, mode, include_images):
    outer, patch = DATASET_PATCH_DIRS[dataset]
    src = source_root / patch if outer is None else source_root / outer / patch
    if not src.is_dir():
        return {"dataset": dataset, "status": "missing", "source": str(src)}

    dst = output_root / dataset
    tissue_files = iter_files(src / "tissue_masks")
    nuclei_files = iter_files(src / "nuclei_masks")
    tissue_by_name = {p.name: p for p in tissue_files}
    nuclei_by_name = {p.name: p for p in nuclei_files}
    paired_names = sorted(set(tissue_by_name) & set(nuclei_by_name))

    counts = {
        "tissue_masks": len(tissue_files),
        "nuclei_masks": len(nuclei_files),
        "paired_masks": len(paired_names),
        "unpaired_tissue_masks": len(set(tissue_by_name) - set(nuclei_by_name)),
        "unpaired_nuclei_masks": len(set(nuclei_by_name) - set(tissue_by_name)),
    }

    for subdir, files_by_name in (("tissue_masks", tissue_by_name), ("nuclei_masks", nuclei_by_name)):
        for name in paired_names:
            file_path = files_by_name[name]
            link_or_copy(file_path, dst / subdir / file_path.name, mode)

    if include_images:
        for subdir in ("images", "conditioning"):
            files = iter_files(src / subdir)
            for file_path in files:
                link_or_copy(file_path, dst / subdir / file_path.name, mode)
            counts[subdir] = len(files)

    for filename in ("metadata.jsonl", "stats.txt"):
        if (src / filename).exists():
            link_or_copy(src / filename, dst / filename, mode)

    return {
        "dataset": dataset,
        "status": "ready",
        "source": str(src),
        "target": str(dst),
        "counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Pathology_edit/edit_datasets for upload")
    parser.add_argument("--source-root", default="../datasets", help="Root containing the original datasets")
    parser.add_argument("--output-root", default="edit_datasets", help="Staging output directory")
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_PATCH_DIRS), help="Datasets to stage")
    parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--include-images", action="store_true", help="Also stage RGB images and conditioning masks")
    parser.add_argument("--prune-unpaired", action="store_true",
                        help="Remove staged tissue/nuclei files that are not in the paired source set")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = (repo_root / args.source_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "mode": args.mode,
        "include_images": args.include_images,
        "datasets": [],
    }
    for dataset in args.datasets:
        item = stage_dataset(source_root, output_root, dataset, args.mode, args.include_images)
        if args.prune_unpaired and item.get("status") == "ready":
            target = Path(item["target"]).resolve()
            if output_root not in target.parents and target != output_root:
                raise RuntimeError(f"Refusing to prune outside output root: {target}")
            for subdir in ("tissue_masks", "nuclei_masks"):
                staged_dir = target / subdir
                if not staged_dir.is_dir():
                    continue
                outer, patch = DATASET_PATCH_DIRS[dataset]
                src = source_root / patch if outer is None else source_root / outer / patch
                paired = (
                    {p.name for p in iter_files(src / "tissue_masks")}
                    & {p.name for p in iter_files(src / "nuclei_masks")}
                )
                for staged_file in iter_files(staged_dir):
                    if staged_file.name not in paired:
                        staged_file.unlink()
        manifest["datasets"].append(item)

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme = output_root / "README_PHASE4_UPLOAD.md"
    readme.write_text(
        "# Phase 4 edit_datasets\n\n"
        "Upload this directory together with the Pathology_edit code on the server.\n\n"
        "Each dataset directory contains the mask layers required by Phase 4:\n"
        "- `tissue_masks/*.png`: unified tissue IDs, 0-15\n"
        "- `nuclei_masks/*.png`: raw nuclei IDs, 0/101-105\n\n"
        "Only tissue/nuclei files with matching names are staged. See `manifest.json` "
        "for source counts and paired sample counts.\n\n"
        "Example server run:\n\n"
        "```bash\n"
        "cd Pathology_edit\n"
        "bash scripts/phase4_probnet_workflow.sh BCSS edit_datasets/BCSS phase4_runs/BCSS\n"
        "```\n",
        encoding="utf-8",
    )

    for item in manifest["datasets"]:
        status = item["status"]
        dataset = item["dataset"]
        counts = item.get("counts", {})
        print(f"{dataset}: {status} {counts}")
    print(f"Manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract UniPath patches from WSI files using coordinates stored in HDF5."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import openslide
from PIL import Image
from tqdm import tqdm


WSI_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UniPath patches at level 0 and resize them for RAG."
    )
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: an images directory next to --h5).",
    )
    parser.add_argument("--patch-size", type=int, default=672)
    parser.add_argument("--resize-size", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-wsis",
        type=int,
        help="Process at most this many WSI groups (for smoke tests).",
    )
    return parser.parse_args()


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def build_wsi_index(wsi_dir: Path) -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}

    def add_path(index: dict[str, Path], key: str, path: Path) -> None:
        existing = index.get(key)
        if existing is None or existing == path:
            index[key] = path
            return

        existing_is_flat = existing.parent == wsi_dir
        path_is_flat = path.parent == wsi_dir
        if existing_is_flat != path_is_flat:
            index[key] = path if not path_is_flat else existing
            return

        raise RuntimeError(
            f"Ambiguous WSI key {key!r}: {existing} and {path}"
        )

    for path in tqdm(
        wsi_dir.rglob("*"),
        desc="Indexing WSI files",
        unit="file",
    ):
        if not path.is_file() or path.suffix.lower() not in WSI_SUFFIXES:
            continue
        for key, index in ((path.name, by_name), (path.stem, by_stem)):
            add_path(index, key, path)

    return {**by_stem, **by_name}


def load_groups(h5_path: Path) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with h5py.File(h5_path, "r") as handle:
        required = {"wsi_id", "position", "sample_key"}
        missing = required.difference(handle.keys())
        if missing:
            raise KeyError(f"Missing H5 field(s): {', '.join(sorted(missing))}")

        wsi_ids = handle["wsi_id"]
        positions = handle["position"]
        if len(wsi_ids) != len(positions):
            raise ValueError("wsi_id and position lengths do not match")

        for wsi_id, position in zip(wsi_ids, positions):
            groups[decode_text(wsi_id)].append(
                (int(position[0]), int(position[1]))
            )
    return groups


def main() -> int:
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")

    h5_path = args.h5.expanduser().resolve()
    wsi_dir = args.wsi_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else h5_path.parent / "images"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_index = build_wsi_index(wsi_dir)
    groups = load_groups(h5_path)
    items = list(groups.items())
    if args.max_wsis is not None:
        items = items[: args.max_wsis]

    saved = 0
    skipped = 0
    missing_wsis: list[str] = []
    failures: list[str] = []

    for wsi_id, coordinates in tqdm(items, desc="Processing WSIs", unit="wsi"):
        wsi_path = wsi_index.get(wsi_id)
        if wsi_path is None:
            missing_wsis.append(wsi_id)
            continue

        try:
            with openslide.OpenSlide(str(wsi_path)) as slide:
                for x, y in coordinates:
                    output_path = output_dir / f"{wsi_id}_{x}_{y}.png"
                    if output_path.exists() and not args.overwrite:
                        skipped += 1
                        continue

                    patch = slide.read_region(
                        (x, y),
                        0,
                        (args.patch_size, args.patch_size),
                    ).convert("RGB")
                    if args.resize_size > 0:
                        patch = patch.resize(
                            (args.resize_size, args.resize_size),
                            Image.Resampling.LANCZOS,
                        )
                    patch.save(output_path)
                    saved += 1
        except Exception as exc:
            failures.append(f"{wsi_id}: {exc}")

    print(
        f"saved={saved} skipped={skipped} "
        f"missing_wsis={len(missing_wsis)} failures={len(failures)}"
    )
    for wsi_id in missing_wsis:
        print(f"MISSING_WSI\t{wsi_id}")
    for failure in failures:
        print(f"FAILED_WSI\t{failure}")

    return 1 if missing_wsis or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

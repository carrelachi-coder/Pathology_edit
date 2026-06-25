from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def bytes_to_gib(value: float) -> float:
    return value / (1024**3)


def gib_to_bytes(value: float) -> int:
    return int(value * (1024**3))


def human_bytes(value: float) -> str:
    gib = bytes_to_gib(value)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{value / (1024**2):.2f} MiB"


def find_images(path: Path, recursive: bool, sample_size: int) -> list[Path]:
    if not path.exists():
        return []
    iterator = path.rglob("*") if recursive else path.iterdir()
    images = [item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort()
    return images[:sample_size]


def average_image_size(path: Path | None, recursive: bool, sample_size: int, fallback_mib: float) -> tuple[float, int]:
    if path is None:
        return fallback_mib * (1024**2), 0
    samples = find_images(path, recursive=recursive, sample_size=sample_size)
    if not samples:
        return fallback_mib * (1024**2), 0
    return sum(item.stat().st_size for item in samples) / len(samples), len(samples)


def df_mounts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        output = subprocess.check_output(["df", "-Pk"], text=True)
        for line in output.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            filesystem = " ".join(parts[:-5])
            blocks, used, available, capacity, mount = parts[-5:]
            rows.append(
                {
                    "filesystem": filesystem,
                    "mount": mount,
                    "available_bytes": int(available) * 1024,
                    "capacity": capacity,
                }
            )
    except Exception:
        usage = shutil.disk_usage("/")
        rows.append(
            {
                "filesystem": "/",
                "mount": "/",
                "available_bytes": usage.free,
                "capacity": "",
            }
        )
    return sorted(rows, key=lambda row: int(row["available_bytes"]), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate which mounted disks can hold a large patch + segmentator-mask run.")
    parser.add_argument("--image-root", type=Path, default=None, help="Directory containing source patches. Used to sample average image size.")
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--avg-image-mib", type=float, default=1.0, help="Fallback average image size when --image-root is absent/unreadable.")
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-masks", choices=["none", "selected", "all"], default="all")
    parser.add_argument("--selected-fraction", type=float, default=0.25, help="Expected selected fraction when --write-masks selected.")
    parser.add_argument("--mask-size-px", type=int, default=512, help="Raw 8-bit mask side length estimate.")
    parser.add_argument("--extra-gib", type=float, default=5.0, help="Extra scratch/log space.")
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    avg_image_bytes, sampled = average_image_size(
        args.image_root,
        recursive=args.recursive,
        sample_size=args.sample_size,
        fallback_mib=args.avg_image_mib,
    )
    image_bytes = args.count * avg_image_bytes if args.copy_images else 0
    if args.write_masks == "all":
        mask_count = args.count
    elif args.write_masks == "selected":
        mask_count = int(args.count * args.selected_fraction)
    else:
        mask_count = 0
    raw_mask_bytes = mask_count * args.mask_size_px * args.mask_size_px
    required_bytes = (image_bytes + raw_mask_bytes + gib_to_bytes(args.extra_gib)) * args.safety_factor

    mounts = []
    for row in df_mounts():
        available = int(row["available_bytes"])
        mounts.append(
            {
                **row,
                "available": human_bytes(available),
                "fits": available >= required_bytes,
            }
        )
    result = {
        "count": args.count,
        "sampled_images": sampled,
        "avg_image_size": human_bytes(avg_image_bytes),
        "copy_images": args.copy_images,
        "write_masks": args.write_masks,
        "estimated_mask_count": mask_count,
        "raw_mask_bytes": human_bytes(raw_mask_bytes),
        "extra_space": human_bytes(gib_to_bytes(args.extra_gib)),
        "safety_factor": args.safety_factor,
        "required": human_bytes(required_bytes),
        "mounts": mounts,
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Estimated required space: {result['required']}")
    print(f"Average image size: {result['avg_image_size']} (sampled {sampled}; fallback if 0)")
    print(f"Raw mask estimate: {result['raw_mask_bytes']} for {mask_count} masks")
    print()
    print(f"{'fits':5s} {'available':>12s} {'capacity':>9s} {'mount'}")
    for row in mounts:
        print(f"{'yes' if row['fits'] else 'no':5s} {row['available']:>12s} {str(row['capacity']):>9s} {row['mount']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

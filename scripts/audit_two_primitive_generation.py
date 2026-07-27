#!/usr/bin/env python3
"""Audit local/global synthesis for the paired 1,200-target U1/U2 bank."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image

from phase3_mask_edit.benchmark.pathokid import sha256_file


BACKENDS = {
    "inpaint": "inpaint",
    "cross-v1": "cross-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--expected-nuclei-root", type=Path, required=True)
    parser.add_argument("--expected-pix2pix-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1200)
    parser.add_argument("--expected-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    failures: list[str] = []
    if len(rows) != args.expected_count:
        failures.append(f"manifest count {len(rows)} != {args.expected_count}")
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if not all(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        failures.append("sample IDs must be non-empty and unique")
    if not args.expected_pix2pix_checkpoint.is_file():
        failures.append(
            f"expected pix2pix checkpoint missing: {args.expected_pix2pix_checkpoint}"
        )

    backend_checked = Counter()
    group_checked = Counter()
    for row in rows:
        sample_id = str(row.get("sample_id") or "unknown")
        group = f"{row.get('primitive')}/{row.get('strength')}"
        for backend, selected_mode in BACKENDS.items():
            try:
                sample_dir = args.generated_root / backend / sample_id
                provenance_path = sample_dir / "utility_generation.json"
                image_path = sample_dir / "generated_image.png"
                if not provenance_path.is_file():
                    raise FileNotFoundError(provenance_path)
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8")
                )
                generation = provenance["generation"]
                if provenance.get("sample_id") != sample_id:
                    raise ValueError("provenance sample_id mismatch")
                if provenance.get("backend") != backend:
                    raise ValueError(
                        f"backend={provenance.get('backend')!r}"
                    )
                if generation.get("selected_mode") != selected_mode:
                    raise ValueError(
                        f"selected_mode={generation.get('selected_mode')!r}"
                    )
                nuclei_path = Path(provenance["target_nuclei_mask"])
                support_path = Path(provenance["generation_change_region"])
                if nuclei_path != Path(row["target_nuclei_mask"]):
                    raise ValueError("target nuclei path differs from manifest")
                if not is_relative_to(nuclei_path, args.expected_nuclei_root):
                    raise ValueError(f"unexpected nuclei root: {nuclei_path}")
                if support_path != Path(row["inpaint_change_region"]):
                    raise ValueError("generation support differs from manifest")
                if support_path.name != "inpaint_change_region.png":
                    raise ValueError(
                        f"unexpected generation support: {support_path.name}"
                    )
                for path in (
                    nuclei_path,
                    support_path,
                    Path(row["target_tissue_mask"]),
                ):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                with Image.open(image_path) as image:
                    if image.size != (args.expected_size, args.expected_size):
                        raise ValueError(f"generated size={image.size}")
                    image.convert("RGB")

                if backend == "cross-v1":
                    cross = generation["cross_v1"]
                    pix2pix = cross["pix2pix_v2"]
                    steering = pix2pix["texture_steering"]
                    if cross.get("loads_uni") is not False:
                        raise ValueError("Cross unexpectedly loads UNI")
                    if cross.get("loads_ip_adapter") is not False:
                        raise ValueError("Cross unexpectedly loads IP-Adapter")
                    if (
                        Path(cross["pix2pix_checkpoint"])
                        != args.expected_pix2pix_checkpoint
                    ):
                        raise ValueError(
                            f"unexpected pix2pix checkpoint: "
                            f"{cross['pix2pix_checkpoint']}"
                        )
                    if int(pix2pix.get("epoch", -1)) != 26:
                        raise ValueError(
                            f"pix2pix epoch={pix2pix.get('epoch')!r}"
                        )
                    if int(pix2pix.get("global_step", -1)) != 214895:
                        raise ValueError(
                            f"pix2pix global_step={pix2pix.get('global_step')!r}"
                        )
                    if steering.get("enabled") is not True:
                        raise ValueError("texture steering is not enabled")
                    if (
                        steering.get("reference_direction_mode")
                        != "local_histogram"
                    ):
                        raise ValueError(
                            "reference direction mode is not local_histogram"
                        )
                    expected_scales = ["1/1", "1/2", "1/4", "1/8", "1/16"]
                    if steering.get("scales") != expected_scales:
                        raise ValueError(
                            f"unexpected steering scales: {steering.get('scales')!r}"
                        )
                backend_checked[backend] += 1
                group_checked[(group, backend)] += 1
            except Exception as exc:
                failures.append(
                    f"{sample_id}/{backend}: {type(exc).__name__}: {exc}"
                )

    report = {
        "schema_version": 1,
        "status": "complete" if not failures else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "expected_count_per_backend": args.expected_count,
        "wsi_count": len({str(row.get("wsi_id") or "") for row in rows}),
        "backend_checked": dict(sorted(backend_checked.items())),
        "group_backend_checked": {
            f"{group}/{backend}": group_checked[(group, backend)]
            for group, backend in sorted(group_checked)
        },
        "expected_nuclei_root": str(args.expected_nuclei_root),
        "expected_pix2pix_checkpoint": str(
            args.expected_pix2pix_checkpoint
        ),
        "expected_pix2pix_sha256": (
            sha256_file(args.expected_pix2pix_checkpoint)
            if args.expected_pix2pix_checkpoint.is_file()
            else None
        ),
        "expected_pix2pix_epoch": 26,
        "expected_pix2pix_global_step": 214895,
        "expected_reference_direction_mode": "local_histogram",
        "failure_count": len(failures),
        "failures": failures[:200],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

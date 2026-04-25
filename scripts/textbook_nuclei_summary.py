"""Build compact cancer-specific nuclei summaries for the textbook benchmark tool."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


_CANCER_KEY_ALIASES = {
    "oral_scc": "oral",
    "oral squamous cell carcinoma": "oral",
}

_TUMOR_TISSUE_IDS_BY_DATASET = {
    "BCSS": {1, 14, 15},
    "PANDA": {8, 9, 10},
    "GLAS": {11, 12, 13},
    "IGNITE": {1},
    "PUMA": {1},
    "ORCA": {1},
}


def summarize_nuclei_library(library_dir: str | Path, target_mpp: float = 0.25) -> dict:
    """Summarize a Phase 4 nuclei library into a small browser-friendly JSON payload."""
    library_dir = Path(library_dir)
    raw_stats = _load_statistics(library_dir)
    dataset = str(raw_stats.get("dataset") or library_dir.name)
    cancer_type = str(raw_stats.get("cancer_type") or dataset.lower())
    library_key = _library_key(cancer_type)
    statistics = raw_stats.get("statistics", {})

    areas_by_type: dict[str, list[float]] = defaultdict(list)
    areas_by_tissue: dict[str, list[float]] = defaultdict(list)
    all_areas: list[float] = []

    instances_dir = library_dir / "nuclei_instances"
    if instances_dir.exists():
        for npz_path in sorted(instances_dir.rglob("*.npz")):
            area, nuc_type = _read_instance_area_and_type(npz_path)
            tissue_id = _parse_tissue_id(npz_path.parent.name)
            all_areas.append(area)
            areas_by_type[str(nuc_type)].append(area)
            if tissue_id is not None:
                areas_by_tissue[str(tissue_id)].append(area)

    density = _weighted_density_per_10k(statistics)
    tumor_density, tumor_neoplastic_fraction = _tumor_density_and_neoplastic_fraction(
        dataset=dataset,
        statistics=statistics,
    )

    stored_count_by_type = {
        nuc_type: len(values) for nuc_type, values in sorted(areas_by_type.items())
    }
    stored_count_by_tissue = {
        tissue_id: len(values) for tissue_id, values in sorted(areas_by_tissue.items())
    }

    summary = {
        "dataset": dataset,
        "cancer_type": cancer_type,
        "library_key": library_key,
        "target_mpp": float(target_mpp),
        "source_library_dir": str(library_dir),
        "stored_instance_count": len(all_areas),
        "stored_instance_count_by_type": stored_count_by_type,
        "stored_instance_count_by_tissue": stored_count_by_tissue,
        "density_per_10k_px_weighted": _round_or_none(density),
        "expected_nuclei_per_512_patch_weighted": _round_or_none(
            None if density is None else density * (512 * 512 / 10000.0)
        ),
        "tumor_density_per_10k_px": _round_or_none(tumor_density),
        "tumor_neoplastic_fraction": _round_or_none(tumor_neoplastic_fraction),
        "type_stats": _type_stats(areas_by_type, target_mpp),
        "tissue_stats": _tissue_stats(statistics),
    }
    summary.update(_flat_area_and_diameter_stats(all_areas, target_mpp, "nucleus"))
    summary.update(_flat_area_and_diameter_stats(areas_by_type.get("101", []), target_mpp, "neoplastic"))
    return summary


def write_nuclei_summary_files(
    library_root: str | Path,
    output_dir: str | Path,
    target_mpp: float = 0.25,
) -> list[Path]:
    """Write one summary JSON per cancer type and return written paths."""
    library_root = Path(library_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for library_dir in sorted(path for path in library_root.iterdir() if path.is_dir()):
        if not (library_dir / "statistics.json").exists():
            continue
        summary = summarize_nuclei_library(library_dir, target_mpp=target_mpp)
        output_path = output_dir / f"{summary['library_key']}.json"
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        written.append(output_path)
    return written


def _load_statistics(library_dir: Path) -> dict:
    stats_path = library_dir / "statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistics.json under {library_dir}")
    return json.loads(stats_path.read_text(encoding="utf8"))


def _read_instance_area_and_type(npz_path: Path) -> tuple[float, int]:
    with np.load(npz_path) as data:
        area = float(data["area"])
        nuc_type = int(data["type"])
    return area, nuc_type


def _parse_tissue_id(bucket_name: str) -> int | None:
    parts = bucket_name.split("_")
    if len(parts) < 2 or parts[0] != "tissue":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _library_key(cancer_type: str) -> str:
    normalized = cancer_type.strip().lower().replace("-", "_").replace(" ", "_")
    return _CANCER_KEY_ALIASES.get(normalized, normalized)


def _flat_area_and_diameter_stats(areas: Iterable[float], target_mpp: float, prefix: str) -> dict:
    values = np.asarray(list(areas), dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_area_px_p25": None,
            f"{prefix}_area_px_median": None,
            f"{prefix}_area_px_p75": None,
            f"{prefix}_diameter_px_p25": None,
            f"{prefix}_diameter_px_median": None,
            f"{prefix}_diameter_px_p75": None,
            f"{prefix}_diameter_um_p25": None,
            f"{prefix}_diameter_um_median": None,
            f"{prefix}_diameter_um_p75": None,
        }

    diameters_px = 2.0 * np.sqrt(values / math.pi)
    diameters_um = diameters_px * float(target_mpp)
    return {
        f"{prefix}_area_px_p25": _percentile(values, 25),
        f"{prefix}_area_px_median": _percentile(values, 50),
        f"{prefix}_area_px_p75": _percentile(values, 75),
        f"{prefix}_diameter_px_p25": _percentile(diameters_px, 25),
        f"{prefix}_diameter_px_median": _percentile(diameters_px, 50),
        f"{prefix}_diameter_px_p75": _percentile(diameters_px, 75),
        f"{prefix}_diameter_um_p25": _percentile(diameters_um, 25),
        f"{prefix}_diameter_um_median": _percentile(diameters_um, 50),
        f"{prefix}_diameter_um_p75": _percentile(diameters_um, 75),
    }


def _type_stats(areas_by_type: dict[str, list[float]], target_mpp: float) -> dict:
    stats = {}
    for nuc_type, areas in sorted(areas_by_type.items()):
        entry = {"stored_count": len(areas)}
        entry.update(_flat_area_and_diameter_stats(areas, target_mpp, "nucleus"))
        stats[nuc_type] = entry
    return stats


def _tissue_stats(statistics: dict) -> dict:
    output = {}
    for tissue_id, info in sorted(statistics.items(), key=lambda item: int(item[0])):
        output[tissue_id] = {
            "name": info.get("name", ""),
            "total_area_pixels": info.get("total_area_pixels", 0),
            "total_nuclei": info.get("total_nuclei", 0),
            "density_per_10k_px": info.get("density_per_10k_px", 0.0),
            "nuclei_types": info.get("nuclei_types", {}),
        }
    return output


def _weighted_density_per_10k(statistics: dict) -> float | None:
    total_area = 0.0
    total_nuclei = 0.0
    for info in statistics.values():
        total_area += float(info.get("total_area_pixels", 0) or 0)
        total_nuclei += float(info.get("total_nuclei", 0) or 0)
    if total_area <= 0:
        return None
    return total_nuclei / (total_area / 10000.0)


def _tumor_density_and_neoplastic_fraction(
    *,
    dataset: str,
    statistics: dict,
) -> tuple[float | None, float | None]:
    tumor_ids = _TUMOR_TISSUE_IDS_BY_DATASET.get(dataset.upper()) or _infer_tumor_tissue_ids(statistics)
    total_area = 0.0
    total_nuclei = 0.0
    neoplastic_count = 0.0
    for tissue_id in tumor_ids:
        info = statistics.get(str(tissue_id))
        if not info:
            continue
        total_area += float(info.get("total_area_pixels", 0) or 0)
        total_nuclei += float(info.get("total_nuclei", 0) or 0)
        neoplastic_count += float(
            info.get("nuclei_types", {}).get("101", {}).get("count", 0) or 0
        )
    density = None if total_area <= 0 else total_nuclei / (total_area / 10000.0)
    fraction = None if total_nuclei <= 0 else neoplastic_count / total_nuclei
    return density, fraction


def _infer_tumor_tissue_ids(statistics: dict) -> set[int]:
    ids = set()
    tumor_words = ("tumor", "gleason", "adenomatous", "differentiated", "carcinoma", "dcis")
    for tissue_id, info in statistics.items():
        name = str(info.get("name", "")).lower()
        if any(word in name for word in tumor_words):
            ids.add(int(tissue_id))
    return ids


def _percentile(values: np.ndarray, percentile: float) -> float:
    return _round_or_none(float(np.percentile(values, percentile)))


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build textbook nuclei library summary JSON files.")
    parser.add_argument(
        "--library-root",
        default="phase4_runs/all6/nuclei_library",
        help="Root containing per-dataset nuclei library directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="textbook_tool_static/public/nuclei_library_stats",
        help="Output directory for browser-facing nuclei summary JSON files.",
    )
    parser.add_argument("--target-mpp", type=float, default=0.25)
    args = parser.parse_args()

    written = write_nuclei_summary_files(
        library_root=args.library_root,
        output_dir=args.output_dir,
        target_mpp=args.target_mpp,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

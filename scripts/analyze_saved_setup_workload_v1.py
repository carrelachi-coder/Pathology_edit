"""Offline metadata arithmetic only: never opens any listed runtime/input path."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def analyze(inventory_path, environment_path):
    invraw = Path(inventory_path).read_bytes()
    envraw = Path(environment_path).read_bytes()
    inv = json.loads(invraw)
    env = json.loads(envraw)
    sizes = []
    groups = defaultdict(lambda: {"files": 0, "bytes": 0})
    for name, entry in inv["files"].items():
        size = entry["bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("invalid saved size")
        group = (
            "arrays"
            if "/array_" in name
            else "event_blocks"
            if "/block_" in name
            else "controls_and_indexes"
        )
        groups[group]["files"] += 1
        groups[group]["bytes"] += size
        sizes.append(size)
    total = sum(sizes)
    ordered = sorted(sizes)
    quantiles = {
        str(q): ordered[max(0, math.ceil(len(ordered) * q) - 1)]
        for q in [0, 0.5, 0.9, 0.95, 0.99, 1]
    }
    for row in groups.values():
        row["byte_fraction"] = row["bytes"] / total
    packages = env["packages"]
    return {
        "scope": "offline arithmetic from saved inventory/environment JSON only; no server connection or listed-path opens",
        "input_inventory_sha256": hashlib.sha256(invraw).hexdigest(),
        "environment_audit_sha256": hashlib.sha256(envraw).hexdigest(),
        "input_files": len(sizes),
        "canonical_input_total_bytes": total,
        "groups": dict(groups),
        "file_size_quantiles_nearest_rank_bytes": quantiles,
        "distribution_count": len(packages),
        "saved_distribution_record_hashes": sum(
            x["record_sha256"] is not None for x in packages
        ),
        "distribution_record_total_bytes": None,
        "distribution_record_size_quantiles": None,
        "missing_record_size_evidence": "Existing audit stores RECORD SHA only, without sizes; SHA cannot recover byte lengths. No target rescan performed.",
        "other_distribution_metadata_bytes": None,
        "other_setup_hash_bytes": None,
        "minimum_input_only_bytes_per_second_for_all_60_seconds": total / 60,
        "rate_interpretation": "Arithmetic workload ratio only, ignoring all other stages, metadata and I/O latency. Not measured speed, predicted duration or grounds to increase budget.",
        "old_setup_last_stage": None,
        "old_timeout_root_cause": None,
        "execution_permitted": False,
        "server_rescans": 0,
        "new_validation_measurements": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--saved-inventory", required=True, type=Path)
    p.add_argument("--saved-environment", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = analyze(a.saved_inventory, a.saved_environment)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

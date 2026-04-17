#!/usr/bin/env python3
"""Emit shell assignments for a Phase 4 generation profile."""

import argparse
import json
import shlex
from pathlib import Path


KEY_MAP = {
    "gamma_values": "PROFILE_GAMMA_VALUES",
    "prob_count_weight": "PROFILE_PROB_COUNT_WEIGHT",
    "density_scale": "PROFILE_DENSITY_SCALE",
    "density_scale_json": "PROFILE_DENSITY_SCALE_JSON",
    "max_density_per_10k": "PROFILE_MAX_DENSITY_PER_10K",
    "max_count_factor": "PROFILE_MAX_COUNT_FACTOR",
    "min_distance_scale": "PROFILE_MIN_DISTANCE_SCALE",
}


def main():
    parser = argparse.ArgumentParser(description="Load a Phase 4 generation profile")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--profile-dir", required=True)
    args = parser.parse_args()

    profile_json = Path(args.profile_json).resolve()
    profile_dir = Path(args.profile_dir).resolve()
    with profile_json.open("r", encoding="utf-8") as f:
        profiles = json.load(f)

    profile = profiles.get(args.dataset, profiles.get(args.dataset.upper()))
    if profile is None:
        profile = profiles["DEFAULT"]

    for json_key, shell_key in KEY_MAP.items():
        value = profile.get(json_key, "")
        if json_key == "density_scale_json" and value:
            value = str((profile_dir / str(value)).resolve())
        print(f"{shell_key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()

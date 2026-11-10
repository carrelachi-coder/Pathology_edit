#!/usr/bin/env python3
"""Synthetic source-only screen of a proposed extra local topology check.

This does not execute or modify a pathology witness and cannot certify a
production mechanism. It checks whether a locally accepted source deletion
changes the frozen whole-mask hole counter on fixed random binary masks.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from phase3_mask_edit_refine.execution import _hole_count
from phase3_mask_edit_refine.topology import (
    _cached_local_component_count,
    _neighbor_pattern_at,
)


def screen(*, seed: int = 101, trials: int = 20_000) -> dict:
    rng = np.random.default_rng(seed)
    current_source_simple_candidates = 0
    extra_check_accepted = 0
    whole_hole_count_changes = 0
    first_change = None
    for trial in range(trials):
        source = rng.random((9, 9)) < rng.uniform(0.35, 0.75)
        before = _hole_count(source)
        for row, col in np.argwhere(source):
            row, col = int(row), int(col)
            pattern = _neighbor_pattern_at(source, row, col, outside=False)
            background = (~pattern) & 255
            if _cached_local_component_count(pattern, 8, False) != 1:
                continue
            if _cached_local_component_count(background, 4, True) != 1:
                continue
            current_source_simple_candidates += 1
            if _cached_local_component_count(background, 8, True) != 1:
                continue
            extra_check_accepted += 1
            after = source.copy()
            after[row, col] = False
            if _hole_count(after) != before:
                whole_hole_count_changes += 1
                if first_change is None:
                    first_change = {
                        "trial": trial,
                        "row": row,
                        "col": col,
                        "before": before,
                        "after": _hole_count(after),
                    }
    return {
        "kind": "source-only-local-hole-guard-synthetic-screen-v1",
        "seed": seed,
        "trials": trials,
        "mask_shape": [9, 9],
        "source_density_interval": [0.35, 0.75],
        "current_source_simple_candidates": current_source_simple_candidates,
        "extra_check_accepted": extra_check_accepted,
        "whole_hole_count_changes": whole_hole_count_changes,
        "first_change": first_change,
        "scope": "synthetic_single_pixel_source_deletions_only_no_target_or_multifront_contract",
        "qualification_updates_permitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--trials", type=int, default=20_000)
    args = parser.parse_args()
    print(json.dumps(screen(seed=args.seed, trials=args.trials), sort_keys=True))

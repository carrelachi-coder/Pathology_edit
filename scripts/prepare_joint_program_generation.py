#!/usr/bin/env python3
"""Export original reference, final masks and cumulative G for a valid program."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3_joint_edit_refine.program_generator_adapter import build_frozen_program_generator_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program-result', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--backend', choices=('auto', 'inpaint', 'cross'), default='auto')
    args = parser.parse_args()
    _, _, manifest = build_frozen_program_generator_inputs(
        args.program_result, output_dir=args.output_dir, dataset=args.dataset,
        prompt=args.prompt, backend=args.backend,
    )
    print(json.dumps({k: manifest[k] for k in ('cumulative_ledger', 'automatic_route', 'selected_route', 'route_selection')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

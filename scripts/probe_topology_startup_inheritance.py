#!/usr/bin/env python3
"""Non-business startup experiment only: no source, planner, witness or H&E."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.limited_topology_startup import capture, digest, native_threads, parent_gate, restore_controlled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--torch-cpu', action='store_true')
    args = parser.parse_args()
    if args.torch_cpu:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('explicit CPU-only environment required')
        import torch
    parent_gate()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = capture()
    (args.output/'private_parent_state.json').write_text(json.dumps(parent))
    rows = []
    for mode in ('raw_fork', 'controlled_postfork_restore'):
        for index in range(2):
            path = args.output/f'{mode}_{index}.json'
            pid = os.fork()
            if pid == 0:
                try:
                    before = capture()
                    after = restore_controlled(parent) if mode != 'raw_fork' else before
                    row = {'mode':mode, 'index':index, 'native_threads':native_threads(),
                           'before':{k:digest(v) for k,v in before.items()},
                           'after':{k:digest(v) for k,v in after.items()},
                           'after_matches_parent':digest(after)==digest(parent),
                           'business_modules_loaded':False}
                    with path.open('x') as out: json.dump(row, out)
                    os._exit(0)
                except BaseException:
                    os._exit(77)
            deadline = time.monotonic() + 20
            while True:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    if os.waitstatus_to_exitcode(status): raise RuntimeError('startup child failed')
                    break
                if time.monotonic() > deadline:
                    os.kill(pid, 9); os.waitpid(pid, 0); raise TimeoutError('startup probe')
                time.sleep(.01)
            rows.append(json.loads(path.read_text()))
    result={'scope':'non-business startup control experiment; not A/B/C',
            'execution_permitted':False, 'witness_calls':0, 'parent_native_threads':native_threads(),
            'torch_preloaded_cpu_only':args.torch_cpu, 'PYTHONHASHSEED':parent['PYTHONHASHSEED'],
            'parent_digests':{k:digest(v) for k,v in parent.items()}, 'children':rows,
            'historical_rng_recovered':False}
    with (args.output/'report.json').open('x') as out: json.dump(result,out,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

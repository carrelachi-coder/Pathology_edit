#!/usr/bin/env python3
"""Bounded source-mask screens with explicit terminal controller receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3_mask_edit.benchmark.experiment_identity import asset_identity


def run_one(command, *, log_path, seconds):
    """Timeout is a controller outcome, never a failed biological measurement."""
    start = time.monotonic()
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    return {'controller_exit_code':code, 'timed_out':timed_out,
            'elapsed_seconds':time.monotonic()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-freeze', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seconds-per-source', type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.seconds_per_source <= 0:
        raise ValueError('invalid resource bounds')
    pool = json.loads(args.source_freeze.read_text())['candidate_sources']
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('probe_manuscript_joint_capacity.py')
    identity = {
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'source_freeze':asset_identity(args.source_freeze),
        'case_templates':asset_identity(args.cases),
        'probe_script':asset_identity(script), 'controller_script':asset_identity(__file__),
        'requested_sources':len(pool), 'workers':args.workers,
        'seconds_per_source':args.seconds_per_source,
        'scope':'source-only topology qualification; no target approval or H&E generation',
    }
    (args.output/'controller_identity.json').write_text(json.dumps(identity,indent=2))

    def screen(index):
        output = args.output/f'source_{index:03d}'
        command = [sys.executable, str(script), '--source-freeze',str(args.source_freeze),
            '--cases',str(args.cases), '--output',str(output), '--start-index',str(index),
            '--limit','1', '--topology-witness']
        receipt = run_one(command, log_path=args.output/f'source_{index:03d}.log',
                          seconds=args.seconds_per_source)
        receipt.update(index=index, reference_id=pool[index]['reference_id'])
        result_path = output/'capacity_screen.json'
        if receipt['controller_exit_code'] == 0 and result_path.is_file():
            records = json.loads(result_path.read_text())
            if len(records) != 1 or records[0]['reference_id'] != receipt['reference_id']:
                raise ValueError('source screen result identity differs')
            receipt['result'] = records[0]
            receipt['result_asset'] = asset_identity(result_path)
        (args.output/f'source_{index:03d}_controller.json').write_text(json.dumps(receipt,indent=2))
        return receipt

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(screen,index) for index in range(len(pool))]
        for future in as_completed(futures):
            completed.append(future.result())
            (args.output/'controller_progress.json').write_text(json.dumps(sorted(completed,key=lambda x:x['index']),indent=2))
            print(json.dumps({'completed':len(completed),'requested':len(pool),
                'topology_pass':sum(r.get('result',{}).get('both_significant_topology_pass',False) for r in completed),
                'timeout':sum(r['timed_out'] for r in completed)}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

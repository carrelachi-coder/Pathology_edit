"""Local synthetic cost attribution. No pathology imports or witness calls.

Profiler timings include profiler overhead and do not predict target S07 timing.
Fresh output only. No acceptance certificate or experiment authorization issued.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostic_storage.observer_adapter_v1 import ObserverAdapter
from scripts.stress_observer_adapter_v1 import identity, globals_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=int, default=100000)
    args = parser.parse_args()
    if not 1 <= args.events <= 500000:
        parser.error('events must be in1..500000')
    args.output.mkdir(exist_ok=False, parents=True)
    initial = globals_sha()
    core = ObserverAdapter(args.output / 'synthetic', mode='hash_only', identity=identity())
    assert core.valid
    profiler = cProfile.Profile()
    start = time.monotonic()
    profiler.enable()
    for i in range(args.events):
        core.sink.record('guard.stage', {'stage': 'source_present', 'row': i % 512, 'col': (i // 512) % 512})
    core.writer._flush()  # Diagnostic-only flush of unchanged writer, no business completion.
    profiler.disable()
    elapsed = time.monotonic() - start
    core.sink.close()
    assert core.valid and not core.sink.failed and core.events == args.events
    assert globals_sha() == initial
    assert not (args.output/'synthetic/codec/observer_complete.json').exists()
    profiler.dump_stats(str(args.output/'profile.pstats'))
    rows = []
    for (file, line, function), (primitive, calls, self_time, cumulative, callers) in pstats.Stats(profiler).stats.items():
        rows.append({'file': file, 'line': line, 'function': function, 'calls': calls, 'self_seconds': self_time, 'cumulative_seconds': cumulative})
    rows.sort(key=lambda r: r['self_seconds'], reverse=True)
    report = {'scope': 'Local profiled synthetic guard.stage only. Includes profiler overhead. No target timing inference.', 'events': args.events, 'elapsed_seconds': elapsed, 'global_rng_unchanged': True, 'capture_complete': False, 'qualification_updates': 0, 'top_self_time': rows[:25], 'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path('diagnostic_storage/guard_blocks_v2.py'), Path('diagnostic_storage/observer_adapter_v1.py')]}}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'events': args.events, 'elapsed_seconds': elapsed, 'top_self_time': rows[:6]}))


if __name__ == '__main__':
    main()

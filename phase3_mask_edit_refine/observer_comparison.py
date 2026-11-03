"""Strict offline evidence checks. No model call, retry or qualification update."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .passive_observer import array_sha256, file_sha256


def _load(directory):
    root = Path(directory)
    complete = json.loads((root / 'observer_complete.json').read_text())
    if complete['summary_sha256'] != file_sha256(root / 'observer_status.json'):
        raise ValueError('summary does not match completed capture')
    saved = json.loads((root / 'observer_status.json').read_text())
    status = saved['status']
    if status.get('schema_version') != 2:
        raise ValueError('unsupported observer summary schema')
    if not (status['valid'] and status['closed'] and status['interpretation_permitted']) or status.get('faults'):
        raise ValueError('invalid or incomplete observer evidence')
    chain = '0' * 64
    events = []
    with (root / 'semantic.jsonl').open(encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            event = json.loads(line)
            if event['sequence'] != i:
                raise ValueError('event sequence gap')
            events.append(event)
            chain = hashlib.sha256((chain + line).encode('utf-8')).hexdigest()
    if not events or len(events) != status['events'] or chain != status['semantic_chain_sha256']:
        raise ValueError('semantic evidence truncated or modified')
    arrays = []
    def walk(value):
        if type(value) is dict:
            if list(value) == ['array']:
                arrays.append(value['array'])
            else:
                for nested in value.values():
                    walk(nested)
        elif type(value) is list:
            for nested in value:
                walk(nested)
    for event in events:
        walk(event)
    if len(arrays) != status['arrays']:
        raise ValueError('semantic array count does not match summary')
    if status['mode'] == 'full_capture' and len(arrays) != len(saved['artifacts']):
        raise ValueError('full capture raster inventory incomplete')
    if status['mode'] == 'hash_only' and saved['artifacts']:
        raise ValueError('hash-only evidence contains unexpected raster artifacts')
    names = [item['file'] for item in saved['artifacts']]
    if len(set(names)) != len(names):
        raise ValueError('duplicate artifact reference')
    if set(p.name for p in root.glob('array_*.npy')) != set(names):
        raise ValueError('unreferenced or missing raster file')
    for index, item in enumerate(saved['artifacts']):
        name = item['file']
        if Path(name).name != name or name != f'array_{index + 1:08d}.npy' or (root / name).is_symlink():
            raise ValueError('artifact path escapes evidence directory')
        if file_sha256(root / name) != item['sha256']:
            raise ValueError('raster artifact modified or missing')
        array = np.load(root / name, allow_pickle=False, mmap_mode='r')
        expected = arrays[index]
        if array.dtype.str != expected['dtype'] or list(array.shape) != expected['shape'] or array_sha256(array) != expected['content_sha256']:
            raise ValueError('raster content does not match semantic transcript')
    return events, status


def compare_observer_runs(hash_directory, full_directory):
    """No post-hoc field exclusions: compare every semantic value and its order."""
    try:
        left, ls = _load(hash_directory)
        right, rs = _load(full_directory)
        if ls['mode'] != 'hash_only' or rs['mode'] != 'full_capture':
            raise ValueError('comparison requires B hash_only and C full_capture')
        if left != right:
            mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            return {'status': 'mismatch_hold', 'first_different_event': mismatch,
                    'qualification_updates_permitted': False}
        return {'status': 'semantic_match', 'events': len(left), 'excluded_semantic_fields': [],
                'qualification_updates_permitted': False,
                'scope': 'B/C semantic transcript only; original A and historical comparison require separately retained business receipts'}
    except Exception as exc:
        return {'status': 'mismatch_hold', 'reason': type(exc).__name__ + ': ' + str(exc),
                'qualification_updates_permitted': False}


def rejection_pixel_sets(events):
    """Offline unique sets per grow invocation, preserving cross-guard overlap.

    Coordinates retain repeated-visit counts separately. No causal loss estimate.
    """
    output = []
    current = None
    for event in events:
        if event['event'] == 'grow.begin':
            current = {'begin_sequence': event['sequence'], 'rejections': {}, 'event_counts': {}}
            output.append(current)
        elif event['event'] in ('guard.reject', 'guard.local_result') and current is not None:
            values = dict(event['values']['mapping'])
            reason = values['reason']
            if reason is not None:
                current['rejections'].setdefault(reason, set()).add((int(values['row']), int(values['col'])))
                current['event_counts'][reason] = current['event_counts'].get(reason, 0) + 1
    return output

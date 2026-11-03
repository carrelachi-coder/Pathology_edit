"""Reviewed-startup candidate utilities; never imports pathology business code.

This is an experimental control proposal, not recovery of historical RNG state.
CPython random's at-fork handler may reseed a child; raw fork is insufficient.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

HASH_WORDS = ('pathology', 'S07', 'U1', 'U2', 'anchor:1', 'interface:2')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def native_threads():
    directory = Path('/proc/self/task')
    return len(list(directory.iterdir())) if directory.is_dir() else None


def capture():
    torch = sys.modules.get('torch')
    torch_state = None
    if torch is not None:
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized; fork candidate disallowed')
        torch_state = torch.default_generator.get_state().tolist()
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch_cpu': torch_state,
            'hash_fingerprint': digest([hash(word) for word in HASH_WORDS]),
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED')}


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


def restore_controlled(state):
    """Explicit post-fork restore, requiring a separately approved decision.

    Covers default Python/NumPy and preloaded Torch CPU only. No discovery or
    replacement of any local Generator. Never changes the inherited hash secret.
    """
    random.setstate(_tuples(state['python']))
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state['torch_cpu'] is not None:
        torch = sys.modules.get('torch')
        if torch is None or torch.cuda.is_initialized():
            raise RuntimeError('Torch availability/CUDA drift')
        torch.default_generator.set_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8))
    elif 'torch' in sys.modules:
        raise RuntimeError('unexpected Torch initialization')
    now = capture()
    if digest(now) != digest(state):
        raise RuntimeError('controlled startup state mismatch')
    return now


def parent_gate(require_linux_threads=True):
    if not hasattr(os, 'fork'):
        raise RuntimeError('fork unavailable')
    if require_linux_threads and native_threads() != 1:
        raise RuntimeError('parent must have exactly one native thread')
    if any(name.startswith(('phase3_', 'inpaint_cells')) for name in sys.modules):
        raise RuntimeError('business modules already imported in parent')
    capture()  # Reject initialized CUDA without creating CUDA generators.

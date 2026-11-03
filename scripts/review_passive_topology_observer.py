#!/usr/bin/env python3
"""Static observer-only diff audit. Never invokes source qualification or a replay."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE = '20ace976e92b7d3e3de091590e4e2d12653f5def'
BUSINESS_FILES = (
    'phase3_mask_edit_refine/execution.py',
    'phase3_mask_edit_refine/topology.py',
    'phase3_joint_edit_refine/candidate_feasibility.py',
    'phase3_joint_edit_refine/tissue_planner.py',
    'phase3_joint_edit_refine/g2_execution_qualification.py',
)
OBSERVER = 'phase3_mask_edit_refine/passive_observer.py'


def _passive_value(node):
    # No eager callbacks, comprehensions, guards, indexing, reductions or RNG draws.
    return isinstance(node, (ast.Name, ast.Constant)) or (
        isinstance(node, ast.Attribute) and _passive_value(node.value))


class StripObservation(ast.NodeTransformer):
    def __init__(self):
        self.calls = []

    def visit_ImportFrom(self, node):
        if node.module in ('phase3_mask_edit_refine.passive_observer', 'passive_observer'):
            assert len(node.names) == 1 and node.names[0].name == 'observe'
            assert node.names[0].asname is None
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'observe':
            assert len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
            assert isinstance(call.args[0].value, str)
            assert all(kw.arg is not None and _passive_value(kw.value) for kw in call.keywords)
            self.calls.append({'line': node.lineno, 'event': call.args[0].value})
            return None
        return self.generic_visit(node)


def review(root):
    rows = []
    for file in BUSINESS_FILES:
        old = subprocess.check_output(['git', 'show', BASE + ':' + file], cwd=root)
        new = (Path(root) / file).read_bytes()
        stripped = StripObservation()
        tree = stripped.visit(ast.parse(new))
        same = ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False)
        rows.append({'file': file, 'base_sha256': hashlib.sha256(old).hexdigest(),
                     'build_sha256': hashlib.sha256(new).hexdigest(),
                     'business_ast_unchanged_after_removing_observer_import_and_calls': same,
                     'passive_calls': stripped.calls})
    return {'scope': 'static review only; no qualification execution', 'base_commit': BASE,
            'files': rows, 'passed': all(r['business_ast_unchanged_after_removing_observer_import_and_calls'] for r in rows),
            'observer_sha256': hashlib.sha256((Path(root) / OBSERVER).read_bytes()).hexdigest(),
            'limitation': 'AST equivalence does not prove runtime noninterference; synthetic tests and separately approved real comparisons remain required.'}


if __name__ == '__main__':
    result = review(Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)

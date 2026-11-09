from collections import Counter
from types import SimpleNamespace

from scripts import run_budget_cost_target_trial_v3 as runner


def test_duplicate_distribution_names_are_distinct(monkeypatch):
    first = SimpleNamespace(metadata={'Name': 'RF-Solver-Editing'}, version='0.0.0', _path='first')
    second = SimpleNamespace(metadata={'Name': 'RF-Solver-Editing'}, version='0.0.0', _path='second')
    monkeypatch.setattr(runner.importlib.metadata, 'distributions', lambda: [first, second])
    monkeypatch.setattr(runner, 'record_sha256', lambda dist: {'first': 'a' * 64, 'second': None}[dist._path])
    rows = [
        {'name': 'RF-Solver-Editing', 'version': '0.0.0', 'record_sha256': 'a' * 64},
        {'name': 'RF-Solver-Editing', 'version': '0.0.0', 'record_sha256': None},
    ]
    assert runner.distribution_inventory() == runner.frozen_distribution_inventory(rows)
    assert runner.distribution_inventory() != runner.frozen_distribution_inventory(rows[:1] * 2)


def test_distribution_inventory_rejects_changed_digest(monkeypatch):
    dist = SimpleNamespace(metadata={'Name': 'pkg'}, version='1', _path='x')
    monkeypatch.setattr(runner.importlib.metadata, 'distributions', lambda: [dist])
    monkeypatch.setattr(runner, 'record_sha256', lambda _: 'a' * 64)
    rows = [{'name': 'pkg', 'version': '1', 'record_sha256': 'b' * 64}]
    assert runner.distribution_inventory() != runner.frozen_distribution_inventory(rows)

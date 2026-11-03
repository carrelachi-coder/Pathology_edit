"""Proposed read-only engineering budget. Not S07/production authorization.

Trusted code selects a bounded task policy; capture input cannot set a deadline.
Legacy producer/validator files remain immutable. Server trial requires review of
its separately frozen measurement request before this candidate is activated.
"""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

POLICY_ID = "readonly-validation-budget-v2-30-60"
COMPATIBILITY_ID = "producer-32dc6007-readonly-primitives-v1"
BUDGETS = MappingProxyType({"capture": 30, "compare": 60})
PRODUCER_PINS = MappingProxyType(
    {
        "guard_blocks_v2.py": "dd4103b00b7eebf2ac4018a7997a61a01826840a0647cc3c9d19dafd53ecbef4",
        "observer_adapter_contract_v1.py": "18b4fcb5e8721df2b3ed835105cb479b920ec91f1b8c1ae65e683648451e7342",
        "observer_adapter_v1.py": "d646b6d072cb45d400c00ab90a352deeaee67db808735a61a06b3d68a2e4d421",
        "observer_adapter_validation_v1.py": "8888495d26574ce6bd9b5f811a0c83c8b5f64cdf4edf7315504310bcb21f8fcf",
    }
)
LEGACY_GATE_PIN = "8f45bd61cb4f5fc1a0b221d3997fe8f92a5dfc64ccf3ca53d1b574dd67601f53"


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def budget_for(kind):
    if type(kind) is not str or kind not in BUDGETS:
        raise ValueError("unknown validation task")
    return BUDGETS[kind]


def policy_identity():
    value = {
        "id": POLICY_ID,
        "version": 2,
        "hard_seconds_by_kind": dict(BUDGETS),
        "automatic_retries": 0,
        "read_only": True,
        "publish": False,
        "compatibility": COMPATIBILITY_ID,
    }
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {"policy_id": POLICY_ID, "policy_sha256": hashlib.sha256(raw).hexdigest()}


def verify_legacy_files():
    root = Path(__file__).parent
    pins = dict(PRODUCER_PINS)
    pins["validator_process_gate_v1.py"] = LEGACY_GATE_PIN
    for name, expected in pins.items():
        if file_sha(root / name) != expected:
            raise ValueError("unreviewed legacy component drift: " + name)
    return dict(PRODUCER_PINS)


def consumer_identity():
    root = Path(__file__).parent
    paths = [
        root / "validation_budget_policy_v2.py",
        root / "observer_adapter_validation_v2.py",
        root / "validator_process_gate_v2.py",
        root.parent / "scripts/validator_readonly_worker_v2.py",
    ]
    return {p.name: file_sha(p) for p in paths}

"""Read-only 30/60-second budget for the pinned Adapter v2 capture format.

Trusted code selects the task budget; capture input cannot set a deadline.
This module does not authorize a source, witness or H&E call.
"""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

POLICY_ID = "readonly-validation-budget-v3-adapter-v2-30-60"
COMPATIBILITY_ID = "producer-adapter-v2-budget-readonly-primitives-v1"
BUDGETS = MappingProxyType({"capture": 30, "compare": 60})
PRODUCER_PINS = MappingProxyType(
    {
        "guard_blocks_v2.py": "dd4103b00b7eebf2ac4018a7997a61a01826840a0647cc3c9d19dafd53ecbef4",
        "guard_budget_candidate_v1.py": "d4ac09d03509e9d149b96e184d0abdebab08462730b7e5a494af7432df03cf64",
        "observer_adapter_contract_v2.py": "76c2ef88254cbffece3cca947546ae67a0c6fbcb17530caef8b2525e41c54188",
        "observer_adapter_v2.py": "35cf0841325b51e809f803fa9fd3346ebd12ea5d6ae04fcaa03c5c3abc2104f7",
        "observer_adapter_validation_v2_budget.py": "4e513943a97bd07b022b7103457e936896a10315612d5353a55cf7745d1faac1",
    }
)


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
        "version": 3,
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


def verify_producer_files():
    root = Path(__file__).parent
    for name, expected in PRODUCER_PINS.items():
        if file_sha(root / name) != expected:
            raise ValueError("new producer component drift: " + name)
    return dict(PRODUCER_PINS)


def consumer_identity():
    root = Path(__file__).parent
    paths = [
        root / "validation_budget_policy_v3_budget.py",
        root / "observer_adapter_validation_v3_budget.py",
        root / "validator_process_gate_v3_budget.py",
        root.parent / "scripts/validator_readonly_worker_v3_budget.py",
    ]
    return {p.name: file_sha(p) for p in paths}

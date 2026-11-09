"""Versioned budget adapter keeps the legacy Codec/Adapter pins intact."""
import pytest

from diagnostic_storage import guard_blocks_v2 as codec
from diagnostic_storage.guard_budget_candidate_v1 import CandidateBudget
from diagnostic_storage import observer_adapter_contract_v1 as legacy_contract
from diagnostic_storage import observer_adapter_contract_v2 as contract
from diagnostic_storage.observer_adapter_v1 import ObserverAdapter as LegacyAdapter
from diagnostic_storage.observer_adapter_v2 import ObserverAdapter


def identity():
    return {
        'reference_id': 'synthetic-source', 'patient_id': 'synthetic-patient',
        'operations': [{'case_id': 'U1', 'primitive_id': 'p1'}, {'case_id': 'U2', 'primitive_id': 'p2'}],
        **{k: '1' * 64 for k in ('run_manifest_sha256', 'input_identity_sha256',
                               'case_mapping_sha256', 'contract_sha256', 'startup_identity_sha256')},
    }


def test_new_identity_binds_candidate_without_changing_legacy():
    assert legacy_contract.code_identity()['guard_blocks_v2.py'] == legacy_contract.CODEC_PIN
    new = contract.code_identity()
    assert new['guard_blocks_v2.py'] == contract.CODEC_PIN
    assert new['guard_budget_candidate_v1.py'] == contract.BUDGET_PIN
    assert 'guard_budget_candidate_v1.py' not in legacy_contract.code_identity()


def test_candidate_hash_mismatch_refuses_new_adapter(monkeypatch, tmp_path):
    original = codec.file_hash
    def changed(path):
        return '0' * 64 if str(path).endswith('guard_budget_candidate_v1.py') else original(path)
    monkeypatch.setattr(codec, 'file_hash', changed)
    with pytest.raises(contract.AdapterError, match='Budget candidate changed'):
        contract.code_identity()
    core = ObserverAdapter(tmp_path / 'rejected', mode='hash_only', identity=identity())
    assert not core.valid
    assert core.writer is None


def test_new_adapter_uses_candidate_while_legacy_stays_original(tmp_path):
    old = LegacyAdapter(tmp_path / 'legacy', mode='hash_only', identity=identity())
    new = ObserverAdapter(tmp_path / 'new', mode='hash_only', identity=identity())
    assert old.valid and new.valid
    assert type(old.writer.budget) is codec.Budget
    assert type(new.writer.budget) is CandidateBudget
    assert old.manifest['format'] != new.manifest['format']
    old.sink.close(); new.sink.close()

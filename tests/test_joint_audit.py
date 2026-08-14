from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from phase3_joint_edit_refine.audit import JointAuditWriter
from phase3_joint_edit_refine.tissue_execution import TissueExecutionBatch


def test_empty_tissue_execution_batch_is_audited_without_rendering(tmp_path):
    writer = JointAuditWriter(tmp_path, case_id="empty-tissue-batch")
    batch = TissueExecutionBatch(
        certified_candidates=(),
        all_candidates=(),
        tissue_gate_reports=(),
        cell_feasibility_reports=(),
        executable_contracts=(),
        executable_contract_errors={},
    )

    result = writer.write_tissue_execution_review(
        pass_index=2,
        source_image_path=str(tmp_path / "unused.png"),
        source_tissue=np.zeros((8, 8), dtype=np.uint8),
        source_nuclei=np.zeros((8, 8), dtype=np.uint8),
        tissue_scene=SimpleNamespace(),
        tissue_plan=SimpleNamespace(candidate_interfaces=()),
        execution_batch=batch,
    )

    assert result is None
    manifest = tmp_path / "empty-tissue-batch" / "tissue_candidates_pass_2.json"
    assert json.loads(manifest.read_text(encoding="utf-8")) == []

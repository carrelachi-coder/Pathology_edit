"""Read-only adapter that restricts frozen ProbNet to legal-point ranking."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import JointContractError


@dataclass
class FrozenProbNetSpatialRanker:
    """Score deterministic template anchors; never chooses counts or shapes."""

    model: Any
    cancer_id: int
    pathology_domain_id: str
    device: Any
    checkpoint_sha256: str
    name: str = "frozen_probnet_spatial_ranker"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        cancer_id: int,
        pathology_domain_id: str,
        device: str = "cpu",
        base_channels: int = 32,
    ) -> FrozenProbNetSpatialRanker:
        if not 0 <= int(cancer_id) <= 5:
            raise JointContractError("ProbNet cancer_id must be in [0, 5]")
        from inpaint_cells.generate import load_checkpoint_model

        path = Path(checkpoint)
        if not path.is_file():
            raise JointContractError(f"ProbNet checkpoint does not exist: {path}")
        model = load_checkpoint_model(str(path), device, base_channels)
        return cls(
            model=model,
            cancer_id=int(cancer_id),
            pathology_domain_id=pathology_domain_id,
            device=device,
            checkpoint_sha256=_sha256(path),
        )

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "adapter": "frozen-probnet-ranker-v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "cancer_id": self.cancer_id,
            "pathology_domain_id": self.pathology_domain_id,
            "role": "legal_template_anchor_ranking_only",
        }

    def score(
        self,
        *,
        tissue_mask: np.ndarray,
        source_nuclei: np.ndarray,
        cell_class: int,
        legal_zone: np.ndarray,
        context: Mapping[str, Any],
    ) -> np.ndarray:
        del context
        if cell_class not in range(1, 6):
            raise JointContractError("ProbNet cell class must use internal ID 1..5")
        from inpaint_cells.generate import predict_fields

        probability, _ = predict_fields(
            self.model,
            np.asarray(tissue_mask),
            np.asarray(source_nuclei),
            np.asarray(legal_zone, dtype=bool),
            self.cancer_id,
            self.device,
        )
        if probability.ndim != 3 or probability.shape[0] <= cell_class:
            raise JointContractError("ProbNet returned an incompatible class probability field")
        score = np.asarray(probability[cell_class], dtype=float)
        if score.shape != np.asarray(legal_zone).shape:
            raise JointContractError("ProbNet probability field is not aligned to the case")
        return score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

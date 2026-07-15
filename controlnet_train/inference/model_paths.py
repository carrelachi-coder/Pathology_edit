"""Canonical production model locations with environment overrides."""

from __future__ import annotations

import os


def _model_path(env_name: str, fallback: str) -> str:
    value = os.environ.get(env_name, "").strip()
    return value or fallback


DEFAULT_INPAINT_CHECKPOINT = _model_path(
    "PATHOLOGY_INPAINT_CHECKPOINT",
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_inpaint_all",
)
DEFAULT_CROSS_V1_CHECKPOINT = _model_path(
    "PATHOLOGY_CROSS_V1_CHECKPOINT",
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1",
)
DEFAULT_PIX2PIX_CHECKPOINT = _model_path(
    "PATHOLOGY_PIX2PIX_CHECKPOINT",
    "/data/wqx/flowedit/"
    "pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft/"
    "ckpt/pilot_step001000.pt",
)
DEFAULT_PROBNET_CHECKPOINT = _model_path(
    "PATHOLOGY_PROBNET_CHECKPOINT",
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/checkpoints/best.pt",
)


__all__ = [
    "DEFAULT_CROSS_V1_CHECKPOINT",
    "DEFAULT_INPAINT_CHECKPOINT",
    "DEFAULT_PIX2PIX_CHECKPOINT",
    "DEFAULT_PROBNET_CHECKPOINT",
]

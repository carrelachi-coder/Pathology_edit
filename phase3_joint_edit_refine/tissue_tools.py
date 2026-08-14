"""Versioned joint tissue tool-family compilation and binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .models import JointContractError

TISSUE_TOOL_MAPPING_VERSION = "joint-tissue-tool-mapping-v2"

JOINT_TOOL_FAMILY_TO_EXECUTOR = {
    "interface_band_sdf": "interface_sdf",
    "topology_safe_morphology": "connected_morphology",
    "organic_v2": "organic_v2",
    "directional_tapered_projection": "directional_tapered_projection",
    "cell_seeded_cord": "cell_seeded_cord",
    "peritumoral_tumor_island": "peritumoral_tumor_island",
}


@dataclass(frozen=True)
class CompiledTissueToolProgram:
    primitive_id: str
    mechanism_id: str
    allowed_joint_families: tuple[str, ...]
    allowed_concrete_executors: tuple[str, ...]
    mapping_version: str = TISSUE_TOOL_MAPPING_VERSION

    @property
    def program_sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_metadata(self) -> dict[str, Any]:
        return {**asdict(self), "program_sha256": self.program_sha256}


def compile_tissue_tool_program(
    *,
    primitive_id: str,
    mechanism_id: str,
    mechanism_allowed_families: tuple[str, ...],
    primitive_allowed_executors: tuple[str, ...],
) -> CompiledTissueToolProgram:
    unknown = sorted(
        set(mechanism_allowed_families) - set(JOINT_TOOL_FAMILY_TO_EXECUTOR)
    )
    if unknown:
        raise JointContractError(
            "joint mechanism contains unmapped tissue tool families: "
            + ", ".join(unknown)
        )
    primitive_allowed = set(primitive_allowed_executors)
    pairs = tuple(
        (family, JOINT_TOOL_FAMILY_TO_EXECUTOR[family])
        for family in mechanism_allowed_families
        if JOINT_TOOL_FAMILY_TO_EXECUTOR[family] in primitive_allowed
    )
    if not pairs:
        raise JointContractError(
            "joint mechanism and mask primitive have no common concrete tissue tool"
        )
    return CompiledTissueToolProgram(
        primitive_id=primitive_id,
        mechanism_id=mechanism_id,
        allowed_joint_families=tuple(family for family, _ in pairs),
        allowed_concrete_executors=tuple(executor for _, executor in pairs),
    )


def validate_tissue_plan_tool_binding(
    plan, *, compiled: CompiledTissueToolProgram
) -> None:
    selected = tuple(plan.tool_program.allowed_tools)
    if not selected or not set(selected).issubset(compiled.allowed_concrete_executors):
        raise JointContractError(
            "tissue plan selected a concrete executor outside the joint mechanism program"
        )
    traced = plan.tool_program.parameter_ranges.get("joint_tissue_tool_program")
    if traced != compiled.to_metadata():
        raise JointContractError(
            "tissue plan is detached from the compiled joint tissue tool program"
        )


def bind_tissue_plan_tool_program(plan, *, compiled: CompiledTissueToolProgram):
    """Narrow a generic mask-tool proposal to the mechanism-owned program.

    Legacy deterministic planners may propose the full mask-primitive tool
    portfolio.  This compiler boundary retains only the concrete executors
    authorized by the active joint mechanism.  A Planner that proposed only
    prohibited tools still fails closed rather than being silently repaired.
    """

    selected = tuple(
        tool
        for tool in plan.tool_program.allowed_tools
        if tool in compiled.allowed_concrete_executors
    )
    if not selected:
        raise JointContractError(
            "tissue plan selected no concrete executor authorized by the joint mechanism"
        )
    from dataclasses import replace

    return replace(
        plan,
        tool_program=replace(
            plan.tool_program,
            allowed_tools=selected,
            parameter_ranges={
                **plan.tool_program.parameter_ranges,
                "joint_tissue_tool_program": compiled.to_metadata(),
            },
        ),
    )

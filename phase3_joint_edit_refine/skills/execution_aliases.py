"""Audited adapters from joint semantics to legacy deterministic tissue tools."""

from __future__ import annotations


# The alias selects an implementation family only. It does not import the
# legacy primitive's pathology meaning: source/target labels, interfaces,
# morphology and postconditions remain owned by the selected joint mechanism.
TISSUE_TOOL_PRIMITIVE_ALIASES = {
    "invasive-front-expansion-v1": "tumor-burden-increase-v1",
}


def tissue_tool_primitive_id(joint_primitive_id: str) -> str:
    return TISSUE_TOOL_PRIMITIVE_ALIASES.get(
        joint_primitive_id, joint_primitive_id
    )

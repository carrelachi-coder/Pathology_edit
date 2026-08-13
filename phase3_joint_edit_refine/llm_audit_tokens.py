"""Closed vocabularies for non-authoritative LLM audit output.

The execution LLM selects typed IDs that already exist in deterministic
certificates. Free prose is neither needed nor accepted because it could imply
unannotated pathology or clinical conclusions. These tokens are neutral audit
labels; compiler-owned rules and metrics retain all decision authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import JointContractError

SEMANTIC_SELECTION_TOKEN = "certified_semantic_option_selected"
SEMANTIC_OBSERVATION_TOKEN = "certified_capability_metrics"
SEMANTIC_CLARIFICATION_TOKEN = "certified_options_require_user_semantic_choice"
SEMANTIC_ABSTAIN_TOKEN = "no_certified_semantic_option"

TISSUE_SELECTION_TOKEN = "certified_tissue_candidate_selected"
CELL_SELECTION_TOKEN = "certified_cell_candidate_selected"
TISSUE_ABSTAIN_TOKEN = "no_certified_tissue_candidate"
CELL_ABSTAIN_TOKEN = "no_certified_cell_candidate"

JOINT_OBSERVATION_TOKEN = "certified_mask_graph_inputs"
JOINT_EXPECTATION_TOKEN = "compiler_owned_render_expectations"
JOINT_ANCHOR_TOKEN = "certified_mask_graph_anchor"
JOINT_ABSTAIN_TOKEN = "joint_contract_not_representable"

CRITIC_SUMMARY_TOKEN = "certified_mask_condition_ranking_completed"
CRITIC_ABSTAIN_SUMMARY_TOKEN = "certified_mask_condition_abstained"
CRITIC_VETO_TOKEN = "mask_condition_contract_veto"


def require_token(value: Any, *, expected: str, field: str) -> str:
    if value != expected:
        raise JointContractError(
            f"{field} must use the compiler-owned neutral audit token"
        )
    return expected


def require_optional_token(
    value: Any,
    *,
    expected: str,
    field: str,
) -> str | None:
    if value is None:
        return None
    return require_token(value, expected=expected, field=field)


def require_exact_tokens(
    values: Any,
    *,
    expected: Sequence[str],
    field: str,
) -> tuple[str, ...]:
    if not isinstance(values, list) or tuple(values) != tuple(expected):
        raise JointContractError(
            f"{field} must use the compiler-owned neutral audit tokens"
        )
    return tuple(expected)


def require_token_subset(
    values: Any,
    *,
    allowed: frozenset[str],
    field: str,
) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(
        isinstance(item, str) and item in allowed for item in values
    ):
        raise JointContractError(
            f"{field} contains non-authoritative free text"
        )
    return tuple(values)

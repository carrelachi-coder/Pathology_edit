#!/usr/bin/env python3
"""Bind one clinician/Codex choice to a joint clarification request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_joint_edit_refine.clarification import (
    create_clarification_decision,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--selected-option-id", required=True)
    parser.add_argument("--responder", required=True)
    parser.add_argument(
        "--provider",
        default="interactive_user_choice",
        choices=("interactive_user_choice", "current_codex_session"),
    )
    parser.add_argument("--rationale")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    decision = create_clarification_decision(
        request,
        selected_option_id=args.selected_option_id,
        responder=args.responder,
        provider=args.provider,
        rationale=args.rationale,
    )
    payload = {
        "case_id": request["case_id"],
        "clarification_decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"case_id": request["case_id"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

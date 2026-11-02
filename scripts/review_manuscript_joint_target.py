#!/usr/bin/env python3
"""Prepare exact mask evidence or consume an already recorded Codex decision."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase3_joint_edit_refine.independent_review import review_evidence
from phase3_joint_edit_refine.approval_handoff import promote_independently_reviewed_joint_candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-case-dir',type=Path,required=True)
    parser.add_argument('--candidate-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--evidence-output',type=Path)
    mode.add_argument('--review',type=Path)
    parser.add_argument('--handoff-output',type=Path)
    args=parser.parse_args()
    if args.evidence_output:
        if args.handoff_output:raise ValueError('evidence preparation does not approve or emit a handoff')
        if args.evidence_output.exists():raise ValueError('preserve prior evidence')
        evidence=review_evidence(args.audit_case_dir,candidate_id=args.candidate_id)
        args.evidence_output.parent.mkdir(parents=True,exist_ok=True)
        args.evidence_output.write_text(json.dumps(evidence,indent=2))
        print(json.dumps({'evidence':str(args.evidence_output),'decision_created':False}))
    else:
        if not args.handoff_output:raise ValueError('review promotion requires --handoff-output')
        paths=promote_independently_reviewed_joint_candidate(args.audit_case_dir,
            candidate_id=args.candidate_id,review_path=args.review,output_dir=args.handoff_output)
        print(json.dumps({'manifest':paths['manifest'],'reviewer':'codex_current_chat'}))
    return 0


if __name__=='__main__':raise SystemExit(main())

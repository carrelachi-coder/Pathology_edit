"""Historical analysis with independently checked patient-sum permutation refits.

Only operation-label permutation computation is substituted. All statistical
rules, draws, repetitions, summaries and plotting inputs use the original code.
"""
from pathlib import Path
import hashlib
import json
import sys
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import analyze_two_primitive_trajectory as analysis
from scripts.trajectory_permutation_candidate_v1 import operation_label_permutation_pvalue


def main():
    args=analysis.parse_args()
    analysis.operation_label_permutation_pvalue=operation_label_permutation_pvalue
    result=analysis.main()
    path=args.output_dir/'two_primitive_trajectory_report.json'
    report=json.loads(path.read_text())
    report['computation']={'method':'patient sums for per-swap leave-one-patient-out direction refits; original arithmetic near boundaries','statistical_rules_changed':False,'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/trajectory_permutation_candidate_v1.py'),Path('scripts/analyze_two_primitive_trajectory.py')]}}
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return result


if __name__=='__main__':
    raise SystemExit(main())

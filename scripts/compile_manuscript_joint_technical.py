#!/usr/bin/env python3
"""Compile real technical targets with mature ProbNet; never auto-approve a critic."""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase3_joint_edit_refine.models import JointCaseContext
from phase3_joint_edit_refine.workflow import JointPathologyEditWorkflow, JointWorkflowConfig
from phase3_joint_edit_refine.tissue_planner import MultiInterfaceResearchTissuePlanner
from phase3_joint_edit_refine.planner import HeuristicJointPlanner
from phase3_joint_edit_refine.critic import DeterministicJointResearchCritic
from phase3_joint_edit_refine.mature_probnet_adapter import MatureProbNetCellExecutor, MatureProbNetConfig
from phase3_joint_edit_refine.probnet_adapter import FrozenProbNetSpatialRanker


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--case-index',type=int,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();raw=json.loads(args.cases.read_text())[args.case_index]; case=JointCaseContext.from_mapping(raw)
    if (args.output/case.case_id).exists(): raise ValueError('preserve prior attempt; choose a new output directory')
    checkpoint='/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt'
    library='/home/lyw/wqx-DL/flow-edit/FlowEdit-main/nuclei_library/BCSS'
    executor=MatureProbNetCellExecutor(MatureProbNetConfig(dataset_name='BCSS',checkpoint=checkpoint,instance_library=library,device=args.device))
    ranker=FrozenProbNetSpatialRanker.from_checkpoint(checkpoint,cancer_id=0,pathology_domain_id=case.pathology_domain_id,device=args.device,base_channels=64)
    result=JointPathologyEditWorkflow(tissue_planner=MultiInterfaceResearchTissuePlanner(),joint_planner=HeuristicJointPlanner(),critic=DeterministicJointResearchCritic(),
        ranker=ranker,cell_executor=executor,config=JointWorkflowConfig(require_mature_probnet_for_target_population_regeneration=True,require_probnet_ranker_for_cell_addition=True)).run(case,output_root=args.output)
    summary={'case_id':case.case_id,'status':result.status,'reasons':result.abstain_reasons,'selected_candidate_id':result.selected_candidate_id,'artifacts':result.artifact_paths}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/(case.case_id+'_summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()

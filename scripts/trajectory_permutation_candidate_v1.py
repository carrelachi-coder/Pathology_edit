"""Independent algebraic candidate for raw patient-label permutation refits.

Fits each held-out direction anew using patient sums, preserving patch weighting.
No change to production analysis. Near-boundary results use original arithmetic.
"""
import itertools
import numpy as np
from scripts.analyze_two_primitive_trajectory import PRIMITIVES, STRENGTHS, unit_rows, cross_fitted_axes


class PreparedPermutation:
    def __init__(self, features, groups, *, strength, backend, selection=None):
        self.features=features;self.groups=np.asarray(groups);self.units,self.ids=np.unique(groups,return_inverse=True)
        self.selected=np.ones(len(groups),dtype=bool) if selection is None else np.asarray(selection,dtype=bool)
        if len(self.units)<2 or self.selected.shape!=self.groups.shape or not self.selected.any():raise ValueError('invalid independent units/selection')
        self.strength,self.backend=strength,backend
        self.sums={}
        for b in ['inpaint','cross']:
            self.sums[b]=[]
            for p in PRIMITIVES:
                delta=features[p]['moderate'][b]-features[p]['moderate']['reference']
                sums=np.zeros((len(self.units),delta.shape[1]),dtype=np.float64)
                np.add.at(sums,self.ids,delta)
                self.sums[b].append(sums)
        diff=unit_rows(features['u1'][strength][backend]-features['u1'][strength]['reference'])-unit_rows(features['u2'][strength][backend]-features['u2'][strength]['reference'])
        self.evaluation=np.zeros((len(self.units),diff.shape[1]),dtype=np.float64)
        self.active=[]
        for g in range(len(self.units)):
            selected=self.selected & (self.ids==g)
            if selected.any():self.evaluation[g]=diff[selected].mean(axis=0);self.active.append(g)
        self.observed=self.raw(np.zeros(len(self.units),dtype=bool))

    def raw(self,swap):
        mask=np.asarray(swap,dtype=bool)[self.ids,None]
        candidate={p:{s:{b:np.where(mask,self.features[other][s][b],v) for b,v in self.features[p][s].items()} for s in STRENGTHS} for p,other in [('u1','u2'),('u2','u1')]}
        axes=cross_fitted_axes(candidate,self.groups)
        d=[unit_rows(candidate[p][self.strength][self.backend]-candidate[p][self.strength]['reference']) for p in PRIMITIVES]
        margin=.5*np.sum((d[0]-d[1])*(axes['u1']-axes['u2']),axis=1)
        return float(np.mean([margin[self.selected & (self.ids==g)].mean() for g in self.active]))

    def score(self,swap):
        swap=np.asarray(swap,dtype=bool)
        fitted=[]
        for p in range(2):
            directions=[]
            for b in ['inpaint','cross']:
                own,other=self.sums[b][p],self.sums[b][1-p]
                grouped=np.where(swap[:,None],other,own)
                held_out=grouped.sum(axis=0)-grouped
                norms=np.linalg.norm(held_out,axis=1)
                # Cancellation-sensitive data retain the original arithmetic.
                if np.any(norms<=1e-8*np.maximum(np.linalg.norm(grouped,axis=1).sum(),1e-12)):
                    return self.raw(swap)
                directions.append(held_out/norms[:,None])
            combined=directions[0]+directions[1]
            if np.any(np.linalg.norm(combined,axis=1)<1e-8):return self.raw(swap)
            fitted.append(unit_rows(combined))
        sign=1.-2.*swap
        margin=.5*np.sum(self.evaluation*(fitted[0]-fitted[1]),axis=1)*sign
        score=float(margin[self.active].mean())
        if abs(score-self.observed)<1e-9:return self.raw(swap)
        return score


def operation_label_permutation_pvalue(features,groups,*,strength,backend,repeats,seed,selection=None):
    prepared=PreparedPermutation(features,groups,strength=strength,backend=backend,selection=selection)
    n=len(prepared.units);exhaustive=n<20 and 2**n<=repeats
    swaps=itertools.product((False,True),repeat=n) if exhaustive else np.random.default_rng(seed).integers(0,2,size=(repeats,n))
    total=2**n if exhaustive else repeats
    exceed=sum(prepared.score(swap)>=prepared.observed-1e-12 for swap in swaps)
    return float(exceed/total if exhaustive else (exceed+1)/(total+1))

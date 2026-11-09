import numpy as np
import pytest
from scripts.analyze_two_primitive_trajectory import operation_label_permutation_pvalue as original
from scripts.trajectory_permutation_candidate_v1 import PreparedPermutation,operation_label_permutation_pvalue as candidate


def fixture(seed,units=5):
    rng=np.random.default_rng(seed);groups=np.repeat(np.arange(units),np.arange(1,units+1));ref=rng.normal(size=(len(groups),9))
    f={p:{s:{'reference':ref.copy(),**{b:ref+rng.normal(size=ref.shape)+rng.normal(size=9) for b in ['inpaint','cross']}} for s in ['moderate','significant']} for p in ['u1','u2']}
    return f,groups


@pytest.mark.parametrize('seed',[1,8,32])
@pytest.mark.parametrize('selection',[False,True])
def test_all_exhaustive_scores_and_pvalues(seed,selection):
    f,g=fixture(seed);sel=np.arange(len(g))%3==0 if selection else None
    for s in ['moderate','significant']:
        for b in ['inpaint','cross']:
            prepared=PreparedPermutation(f,g,strength=s,backend=b,selection=sel)
            for mask in range(32):
                swap=np.array([(mask>>k)&1 for k in range(5)],dtype=bool)
                assert prepared.score(swap)==pytest.approx(prepared.raw(swap),abs=1e-12)
            assert candidate(f,g,strength=s,backend=b,repeats=40,seed=17,selection=sel)==original(f,g,strength=s,backend=b,repeats=40,seed=17,selection=sel)


def test_monte_carlo_same_draws_and_extreme_scale():
    f,g=fixture(991,units=7)
    for p in f:
        for s in f[p]:
            for b in f[p][s]:f[p][s][b]*=1e9
    assert candidate(f,g,strength='significant',backend='cross',repeats=100,seed=19)==original(f,g,strength='significant',backend='cross',repeats=100,seed=19)

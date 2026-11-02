"""Recovery geometry and content identity must not change research eligibility."""
import numpy as np
from scripts.scan_bcss_native_windows import windows, neighbors, digest, cheap


def test_only_complete_origin_locked_windows_and_overlap_references():
    assert windows(511, 900) == []
    assert windows(800, 770) == [(0,0), (0,256), (256,0), (256,256)]
    refs = neighbors('roi',0,0,800,770)
    assert {r['overlap_pixels'] for r in refs} == {131072,65536}
    assert {r['window_id'] for r in refs} == {'roi:y0:x256','roi:y256:x0','roi:y256:x256'}


def test_content_identity_binds_shape_and_all_three_planes():
    a=np.arange(12,dtype=np.uint8)
    assert digest(a.reshape(3,4)) != digest(a.reshape(4,3))
    assert digest(a.reshape(3,4)) == digest(a.reshape(3,4).copy())
    modified=a.reshape(3,4).copy();modified[0,0]=99
    assert digest(modified) != digest(a.reshape(3,4))


def test_exact_stroma_floor_all_failure_reasons_and_legacy_nuclei_ids():
    tissue=np.zeros((512,512),np.uint8);nuclei=np.zeros_like(tissue)
    tissue.flat[:70778]=2;tissue.flat[70778]=1;tissue.flat[70779]=4
    assert cheap(tissue,nuclei)==['stroma_below_27_percent']
    tissue.flat[70780]=2
    assert cheap(tissue,nuclei)==[]  # No additional historical BG<30% filter.
    nuclei[0,0]=255;tissue[0,0]=255
    assert set(cheap(tissue,nuclei))=={'unknown_tissue_id','unknown_nucleus_id','stroma_below_27_percent'}

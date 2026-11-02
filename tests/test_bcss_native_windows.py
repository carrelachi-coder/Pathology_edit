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


def test_invalid_historical_png_is_not_repaired_or_marked_pixel_verified(tmp_path):
    from PIL import Image
    from scripts.scan_bcss_native_windows import index_old, conservative_native_aliases, sha, REMAP
    root=tmp_path/'raw';out=tmp_path/'audit';out.mkdir()
    name='TCGA-AA-0001_x0_y0_py0_px0.png'
    native='TCGA-AA-0001-DX1_xmin0_ymin0_MPP-0.2500.png'
    record={'raw_name':native,'inputs':{}}
    for key,folder,old_folder,array in [('rgb','rgbs','images',np.full((512,512,3),9,np.uint8)),
        ('tissue','masks','tissue_masks',np.ones((512,512),np.uint8)),
        ('nuclei','nuclei_masks','nuclei_masks',np.zeros((512,512),np.uint8))]:
        (root/folder).mkdir(parents=True,exist_ok=True)
        path=root/folder/native;Image.fromarray(array).save(path)
        record['inputs'][key]={'sha256':sha(path)}
        (root/'BCSS_PATCHES'/old_folder).mkdir(parents=True)
        old=root/'BCSS_PATCHES'/old_folder/name
        if key=='rgb':old.write_bytes(b'invalid-PNG')
        else:Image.fromarray(array).save(old)
    coords,content=index_old(root,{name},out)
    assert not content
    lut=np.zeros(256,np.uint8);lut[:22]=REMAP
    aliases=conservative_native_aliases(root,[record],coords,content,lut,out)
    assert len(aliases)==1 and aliases[0]['old_pixel_identity_verified'] is False
    assert (root/'BCSS_PATCHES/images'/name).read_bytes()==b'invalid-PNG'
    assert content  # Native identity conservatively excludes matching future crops.

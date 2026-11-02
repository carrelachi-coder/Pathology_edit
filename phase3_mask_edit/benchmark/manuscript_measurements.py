"""Fixed-source ROI readouts; missing detections can never become measured zero."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scripts.run_cellvit_single_patch import _cellvit_contour_for_single_patch
from .experiment_identity import sha256_file

CELL_POINT_POLICY = 'UI_localized_polygon_area_centroid_else_offset_corrected_native_centroid; floor_xy_into_fixed_roi; exclude_outside_frame'


def tissue_readout(mask, roi, parent):
    mask=np.asarray(mask);roi=np.asarray(roi,dtype=bool)
    if mask.shape!=roi.shape or mask.ndim!=2 or not roi.any():
        raise ValueError('invalid fixed measurement frame or empty source ROI')
    if not np.isin(mask,list(range(8))+[255]).all():raise ValueError('unknown coarse prediction label')
    denominator=int(roi.sum())
    if np.any(mask[roi]==255):
        return {'status':'evaluator_uncertain','value':None,'count':None,'denominator_pixels':denominator}
    count=int(np.count_nonzero(mask[roi]==parent))
    return {'status':'valid_nonzero' if count else 'valid_zero_with_completion',
        'value':count/denominator,'count':count,'denominator_pixels':denominator}


def verify_tissue_prediction(directory, image, *, checkpoint_sha256, release_sha256, code_commit):
    directory=Path(directory);path=directory/'coarse_mask.png'
    proof=json.loads((directory/'provenance.json').read_text())
    if (proof.get('input_image_sha256')!=sha256_file(image)
            or proof.get('checkpoint_sha256')!=checkpoint_sha256
            or proof.get('release_sha256')!=release_sha256
            or proof.get('code_commit')!=code_commit
            or proof.get('code_worktree_dirty') is not False
            or proof.get('profile')!='BCSS'
            or proof.get('output_sha256',{}).get('coarse_mask')!=sha256_file(path)):
        raise ValueError('tissue prediction completion identity differs')
    mask=np.asarray(Image.open(path))
    with Image.open(image) as source:shape=(source.height,source.width)
    if mask.shape!=shape or not np.isin(mask,range(8)).all():
        raise ValueError('invalid completed coarse prediction')
    return mask


def _cell_point(cell, metadata, width, height):
    contour=np.asarray(_cellvit_contour_for_single_patch(cell,width=width,height=height,metadata=metadata),dtype=float)
    if contour.ndim==2 and contour.shape[0]>=3 and contour.shape[1]==2 and np.isfinite(contour).all():
        x,y=contour.T;next_x=np.roll(x,-1);next_y=np.roll(y,-1)
        cross=x*next_y-next_x*y;area2=cross.sum()
        if abs(area2)>1e-8:
            return np.array([np.sum((x+next_x)*cross),np.sum((y+next_y)*cross)])/(3*area2)
    centroid=np.asarray(cell.get('centroid',[]),dtype=float)
    if centroid.shape!=(2,) or not np.isfinite(centroid).all():return None
    if cell.get('offset_global') is not None:
        offset=np.asarray(cell['offset_global'],dtype=float)
        if offset.shape!=(2,) or not np.isfinite(offset).all():return None
        patch=float(metadata.get('patch_size',max(width,height)))
        centroid=centroid-np.array([max(patch-width,0)+offset[1],max(patch-height,0)+offset[0]])
    return centroid if np.isfinite(centroid).all() else None


def inflammatory_density(canonical, mask_path, image, roi, *, model_sha256, mpp):
    canonical=Path(canonical);roi=np.asarray(roi,dtype=bool)
    if roi.ndim!=2 or not roi.any() or not np.isfinite(mpp) or mpp<=0:
        raise ValueError('invalid density ROI or physical scale')
    proof=json.loads(canonical.with_suffix('.completion.json').read_text())
    expected={'json_sha256':sha256_file(canonical),'mask_sha256':sha256_file(mask_path),
        'input_sha256':sha256_file(image),'model_sha256':model_sha256}
    if proof!=expected:raise ValueError('cell detection lacks matching independent completion')
    payload=json.loads(canonical.read_text());provenance=payload.get('benchmark_provenance',{})
    if (provenance.get('status')!='completed'
            or provenance.get('image_sha256')!=expected['input_sha256']
            or provenance.get('model_sha256')!=model_sha256
            or provenance.get('mask_sha256')!=expected['mask_sha256']
            or provenance.get('mpp')!=mpp):
        raise ValueError('cell detection input/model/scale identity differs')
    with Image.open(image) as source:shape=(source.height,source.width)
    cells=payload.get('cells');mask=np.asarray(Image.open(mask_path))
    if not isinstance(cells,list) or roi.shape!=shape or mask.shape!=shape:
        raise ValueError('missing cells or inconsistent detection frame')
    if not np.isin(mask,[0,1,2,3,4,5,101,102,103,104,105]).all():
        raise ValueError('unknown nuclear mask label')
    if not cells and np.any(mask):raise ValueError('empty detections contradict the completed nuclear raster')
    metadata=payload.get('wsi_metadata') or {};count=0;uncertain=0;outside=0;all_type2=0
    for cell in cells:
        if (not isinstance(cell,dict) or type(cell.get('type')) is not int
                or cell['type'] not in range(6)):
            raise ValueError('missing or unknown detected nuclear class')
        if cell['type']!=2:continue
        all_type2+=1
        try:point=_cell_point(cell,metadata,shape[1],shape[0])
        except (ValueError,TypeError,IndexError):point=None
        if point is None:uncertain+=1;continue
        x,y=point
        if not (0<=x<shape[1] and 0<=y<shape[0]):outside+=1;continue
        count+=int(roi[int(np.floor(y)),int(np.floor(x))])
    area_mm2=float(roi.sum())*(mpp/1000.)**2
    return {'status':'evaluator_uncertain' if uncertain else ('valid_nonzero' if count else 'valid_zero_with_completion'),
        'value':None if uncertain else count/area_mm2,'count':None if uncertain else count,
        'resolved_in_roi_count':count,'type2_raw_candidates':all_type2,
        'unlocalizable_type2':uncertain,'outside_frame_type2':outside,
        'total_detected_cells':len(cells),'denominator_pixels':int(roi.sum()),
        'area_mm2_at_nominal_mpp':area_mm2,'nominal_mpp':mpp,'point_policy':CELL_POINT_POLICY}


def verify_postprocess(info, row, *, expected_config):
    """Check actual eligibility, nuclei exclusion and applied alpha rasters."""
    protection=info['cross_v1']['pix2pix_v2']['cross_rgb_od_low_stain_protection']
    if protection.get('configuration')!=expected_config or protection.get('policy')!=expected_config['policy']:
        raise ValueError('postprocess configuration differs from frozen protocol')
    eligibility=np.asarray(Image.open(row['postprocess_eligibility_region']))>0
    nuclei=np.asarray(Image.open(row['target_nuclei_mask']))>0
    alpha=np.asarray(Image.open(protection['mask']))
    radius=protection['nuclei_exclusion_radius_px']
    if (eligibility.ndim!=2 or not eligibility.all() or nuclei.shape!=eligibility.shape
            or alpha.shape!=eligibility.shape or type(radius) is not int or radius<1):
        raise ValueError('postprocess mask frames or exclusion radius differ')
    if radius!=max(1,int(round(expected_config['nuclei_exclusion_radius_at_512']*eligibility.shape[1]/512.))):
        raise ValueError('postprocess exclusion radius differs from frozen configuration')
    digest=lambda array:hashlib.sha256(np.asarray(array,dtype=np.uint8).tobytes()).hexdigest()
    if (digest(eligibility)!=protection['eligibility_region_sha256']
            or digest(alpha)!=protection['applied_mask_sha256']
            or digest(binary_dilation(nuclei,iterations=radius))!=protection['target_nuclei_exclusion_sha256']
            or bool(np.any(alpha))!=protection['applied']
            or protection.get('enabled') is not True):
        raise ValueError('actual postprocess raster identity differs')
    return {**protection,'alpha_nonzero_pixels':int(np.count_nonzero(alpha)),
        'alpha_file_sha256':sha256_file(protection['mask'])}

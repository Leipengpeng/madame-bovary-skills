"""Deterministic scene-referenced grading; no generation, warping, or relighting.

Requires Pillow, NumPy and OpenCV. Run: python match_scene_tone.py --config config.json
The config owns manually reviewed exclusions. Candidates always start from input.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_srgb(path):
    with Image.open(path) as im:
        im.load()
        icc = im.info.get('icc_profile')
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        if icc:
            rgb = ImageCms.profileToProfile(rgb, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                           ImageCms.createProfile('sRGB'), outputMode='RGB')
        return np.array(rgb), 'embedded_ICC_converted_to_sRGB' if icc else 'untagged_assumed_sRGB'


def lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255., cv2.COLOR_RGB2LAB)


def boxes_mask(shape, boxes, inside=False):
    h, w = shape[:2]
    mask = np.full((h, w), not inside, bool)
    for x0, y0, x1, y1 in boxes:
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('Boxes must be normalized, non-empty and inside the image.')
        mask[int(y0*h):int(np.ceil(y1*h)), int(x0*w):int(np.ceil(x1*w))] = inside
    return mask


def geometry_check(ref, src, mask):
    sift = cv2.SIFT_create(nfeatures=5000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), mask.astype(np.uint8)*255)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), mask.astype(np.uint8)*255)
    if d1 is None or d2 is None:
        raise ValueError('Not enough visible background texture; manual registration is needed.')
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .72*p[1].distance]
    if len(good) < 30:
        raise ValueError('Too few background correspondences; refusing color fit.')
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    _, valid = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
    if valid is None or valid.sum() < 30:
        raise ValueError('Background geometry could not be verified.')
    valid = valid[:, 0].astype(bool)
    delta = np.linalg.norm(a[valid]-b[valid], axis=1)/np.hypot(*mask.shape)
    # A homography is estimated only to reject mismatches. No image is warped.
    if np.median(delta) > .003 or np.percentile(delta, 90) > .008:
        raise ValueError('Visible background moved: needs explicit alignment/overlap mask before grading.')
    span = np.ptp(a[valid], axis=0) / np.array([mask.shape[1], mask.shape[0]])
    if min(span) < .45:
        raise ValueError('Matches cover too little of the background to verify the common framing.')
    return {'inliers':int(valid.sum()), 'direct_displacement_percent_diagonal_p50_p90':
            (np.percentile(delta,[50,90])*100).round(4).tolist(), 'no_image_warp':True}


def isotonic(y, weight):
    blocks = []
    for i, (v, w) in enumerate(zip(y, weight)):
        blocks.append([i,i,float(w),float(v)*w])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2] > blocks[-1][3]/blocks[-1][2]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append([a[0], b[1], a[2]+b[2], a[3]+b[3]])
    result = np.empty(len(y),np.float64)
    for start,end,w,total in blocks:
        result[start:end+1] = total/w
    return result


def fit_curve(x, y):
    edges = np.unique(np.quantile(x, np.linspace(0,1,25)))
    xx, yy, ww = [0.], [0.], [len(x)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (x>=lo) & (x<hi)
        if take.sum() >= 25:
            xv = float(np.median(x[take]))
            if xv > xx[-1]+.05 and xv < 99.95:
                xx.append(xv)
                yy.append(float(np.median(y[take])))
                ww.append(int(take.sum()))
    xx.append(100.); yy.append(100.); ww.append(len(x))
    xx, yy = np.array(xx), isotonic(yy,ww)
    yy = np.clip(yy,xx-8,xx+8)
    yy[0], yy[-1] = 0.,100.
    # Reachability bounds keep both endpoints feasible during slope projection.
    for i in range(1,len(xx)):
        dx=xx[i]-xx[i-1]
        lo=max(yy[i-1]+.35*dx,100.-2.*(100.-xx[i]))
        hi=min(yy[i-1]+2.*dx,100.-.35*(100.-xx[i]))
        yy[i]=np.clip(yy[i],lo,hi)
    assert abs(yy[0])<1e-6 and abs(yy[-1]-100)<1e-6
    return xx,yy


def fit_chroma(a, b, scale_enabled):
    # Shared scale preserves chroma angles before the small neutral offset.
    design=np.zeros((len(a)*2,3),np.float64)
    design[0::2,0]=a[:,0]; design[1::2,0]=a[:,1]
    design[0::2,1]=1; design[1::2,2]=1
    target=b.reshape(-1)
    params=np.array([1.,0.,0.])
    weight=np.ones(len(a))
    for _ in range(5):
        wr=np.repeat(weight,2)
        ridge=np.diag([len(a)*3.,len(a)*.02,len(a)*.02])
        lhs=design.T@(wr[:,None]*design)+ridge
        rhs=design.T@(wr*target)+ridge@np.array([1.,0.,0.])
        params=np.linalg.solve(lhs,rhs)
        if not scale_enabled:
            params[0]=1.
            params[1:]=np.median(b-a,axis=0)
        params[0]=np.clip(params[0],.80,1.15)
        params[1:]=np.clip(params[1:],-3.,3.)
        residual=np.linalg.norm(a*params[0]+params[1:]-b,axis=1)
        weight=np.minimum(1.,2.5/np.maximum(residual,.001))
    return params


def apply_transform(img_lab, curve, chroma, strength):
    out=img_lab.copy()
    out[...,0] += strength*(np.interp(img_lab[...,0],*curve)-img_lab[...,0])
    out[...,1:] += strength*(img_lab[...,1:]*chroma[0]+chroma[1:]-img_lab[...,1:])
    return out


def linear_rgb(ll):
    fy=(ll[...,0]+16.)/116.
    ff=np.stack([fy+ll[...,1]/500.,fy,fy-ll[...,2]/200.],axis=-1)
    xyz=np.where(ff>6/29,ff**3,3*(6/29)**2*(ff-4/29))
    xyz*=np.array([.95047,1.,1.08883])
    return xyz@np.array([[3.2404542,-.969266,.0556434],[-1.5371385,1.8760108,-.2040259],[-.4985314,.041556,1.0572252]])


def to_rgb(ll):
    # Compress only out-of-gamut chroma at fixed L*, instead of clipping a channel.
    linear=linear_rgb(ll)
    invalid=np.any((linear<0)|(linear>1),axis=-1)
    result=ll.copy()
    if invalid.any():
        values=ll[invalid].copy()
        lo,hi=np.zeros(len(values)),np.ones(len(values))
        for _ in range(11):
            mid=(lo+hi)*.5
            trial=values.copy(); trial[:,1:]*=mid[:,None]
            rgb=linear_rgb(trial)
            # Leave enough room for 8-bit quantization to avoid a new 0/255 plateau.
            # If neutral L* cannot fit this interior, the later clipping gate rejects it.
            ok=np.all((rgb>=(1/255)/12.92)&(rgb<=((254/255+.055)/1.055)**2.4),axis=-1)
            lo=np.where(ok,mid,lo); hi=np.where(ok,hi,mid)
        values[:,1:]*=lo[:,None]*.999
        result[invalid]=values
    linear=np.clip(linear_rgb(result),0,1)
    encoded=np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)
    return np.rint(encoded*255).astype(np.uint8),float(np.mean(invalid))


def smooth_grid(arr):
    kernel=np.array([1,4,6,4,1],float)/16
    for axis in range(3):
        arr=np.apply_along_axis(lambda x:np.convolve(np.pad(x,2,mode='edge'),kernel,mode='valid'),axis,arr)
    return arr


def lut_lookup(values, lut):
    shape=values.shape
    flat=values.reshape(-1,3)
    index=(flat-np.array([0.,-128.,-128.]))/np.array([100.,256.,256.])*32
    index=np.clip(index,0,32-1e-6)
    lower=np.floor(index).astype(int); frac=index-lower
    answer=np.zeros_like(flat)
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                weight=np.prod(np.where([i,j,k],frac,1-frac),axis=1)
                answer+=weight[:,None]*lut[lower[:,0]+i,lower[:,1]+j,lower[:,2]+k]
    return answer.reshape(shape)


def fit_residual_lut(samples, target, curve, chroma, base_strength, skin_anchors):
    index=(samples-np.array([0.,-128.,-128.]))/np.array([100.,256.,256.])*32
    index=np.clip(index,0,32-1e-6)
    lower=np.floor(index).astype(int); frac=index-lower
    base=apply_transform(samples,curve,chroma,base_strength)
    residual=np.clip(target-base,[-6,-6,-6],[6,6,6])
    sums=np.zeros((33,33,33,3)); counts=np.zeros((33,33,33))
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                weight=np.prod(np.where([i,j,k],frac,1-frac),axis=1)
                ids=(lower[:,0]+i,lower[:,1]+j,lower[:,2]+k)
                np.add.at(counts,ids,weight)
                for ch in range(3):
                    np.add.at(sums[...,ch],ids,weight*residual[:,ch])
    smoothed_counts=smooth_grid(counts)
    correction=np.stack([smooth_grid(sums[...,ch])/(smoothed_counts+1.) for ch in range(3)],axis=-1)
    grid=np.stack(np.meshgrid(np.linspace(0,100,33),np.linspace(-128,128,33),np.linspace(-128,128,33),indexing='ij'),axis=-1)
    # Protect skin-like colors everywhere; this is not a differently graded face mask.
    for anchor in skin_anchors:
        distance=np.sum(((grid-anchor)/np.array([7.,9.,9.]))**2,axis=-1)
        correction*=1-.90*np.exp(-.5*distance)[...,None]
    return grid,correction


def complete_lut(grid, correction, curve, chroma, base_strength, residual_strength):
    mapped=apply_transform(grid,curve,chroma,base_strength)+residual_strength*correction
    # Every fixed a*,b* column is monotonically increasing in L*.
    for j in range(33):
        for k in range(33):
            vals=isotonic(mapped[:,j,k,0],np.ones(33))
            vals[0]=0.; vals[-1]=100.
            for i in range(1,33):
                lo=max(vals[i-1]+.2*3.125,100.-2.5*(100.-i*3.125))
                hi=min(vals[i-1]+2.5*3.125,100.-.2*(100.-i*3.125))
                vals[i]=np.clip(vals[i],lo,hi)
            mapped[:,j,k,0]=vals
    return mapped.astype(np.float32)


def errors(ref, out):
    diff=out-ref
    de=np.linalg.norm(diff,axis=-1)
    return {'mean_abs_L':float(np.mean(np.abs(diff[:,0]))),
            'median_abs_L':float(np.median(np.abs(diff[:,0]))),
            'p90_abs_L':float(np.percentile(np.abs(diff[:,0]),90)),
            'median_a_b_bias':np.median(diff[:,1:],axis=0).tolist(),
            'median_deltaE76':float(np.median(de)), 'p90_deltaE76':float(np.percentile(de,90)),
            'selection_score':float(np.median(de)+.25*np.percentile(de,90))}


def clip_stats(rgb):
    return {'all_channels_near_black_fraction':float(np.mean(np.max(rgb,axis=-1)<=1)),
            'all_channels_near_white_fraction':float(np.mean(np.min(rgb,axis=-1)>=254)),
            'any_channel_endpoint_fraction':float(np.mean(np.any((rgb<=0)|(rgb>=255),axis=-1)))}


def describe(rgb, mask):
    rgb=rgb[mask]
    if not len(rgb):
        return None
    y=rgb@np.array([.2126,.7152,.0722])
    ll=lab(rgb.reshape(-1,1,3)).reshape(-1,3)
    return {'display_luma_p1_p5_p50_p95_p99':np.percentile(y,[1,5,50,95,99]).tolist(),
            'median_rgb':np.median(rgb,axis=0).tolist(),
            'median_Lab_chroma':float(np.median(np.linalg.norm(ll[:,1:],axis=1))),
            **clip_stats(rgb)}


def balanced_indices(mask):
    train,hold=[],[]
    h,w=mask.shape
    # Fixed spatial blocks; holdout blocks are never used to estimate parameters.
    for row in range(9):
        for col in range(12):
            y0,y1=round(row*h/9),round((row+1)*h/9)
            x0,x1=round(col*w/12),round((col+1)*w/12)
            yy,xx=np.where(mask[y0:y1,x0:x1])
            ids=(yy+y0)*w+(xx+x0)
            if len(ids)>250:
                ids=ids[np.linspace(0,len(ids)-1,250).astype(int)]
            (hold if (3*row+5*col)%7==0 else train).extend(ids.tolist())
    if len(train)<1500 or len(hold)<300:
        raise ValueError('Insufficient spatially distributed fit/holdout background.')
    return np.array(train),np.array(hold)


def process_scene(scene, outdir):
    before_hash={p:sha256(p) for p in [scene['reference'],scene['input']]}
    ref,ref_profile=read_srgb(scene['reference'])
    src,src_profile=read_srgb(scene['input'])
    h,w=src.shape[:2]
    if abs((ref.shape[1]/ref.shape[0])/(w/h)-1)>.005:
        raise ValueError('Aspect ratios differ; explicit overlap registration is required.')
    size=(min(w,960),round(h*min(w,960)/w))
    rr=cv2.resize(ref,size,interpolation=cv2.INTER_AREA)
    ss=cv2.resize(src,size,interpolation=cv2.INTER_AREA)
    valid=boxes_mask(rr.shape,scene['exclude_boxes'])
    valid=cv2.erode(valid.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
    geo=geometry_check(rr,ss,valid)
    rlab=lab(cv2.GaussianBlur(rr,(0,0),1.5))
    slab=lab(cv2.GaussianBlur(ss,(0,0),1.5))
    valid &= (rlab[...,0]>2)&(rlab[...,0]<98)&(slab[...,0]>2)&(slab[...,0]<98)
    train,hold=balanced_indices(valid)
    ra,sa=rlab.reshape(-1,3),slab.reshape(-1,3)
    curve=fit_curve(sa[train,0],ra[train,0])
    original_score=errors(ra[hold],sa[hold])
    full_lab=lab(src)
    skin=boxes_mask(src.shape,scene.get('skin_boxes',[]),inside=True)
    original_clips=clip_stats(src)
    aligned_ref=cv2.resize(ref,(w,h),interpolation=cv2.INTER_AREA)
    audit_masks={name:boxes_mask(src.shape,[box],inside=True) for name,box in scene.get('audit_regions',{}).items()}
    audit_clip_limits={name:max(clip_stats(src[m])['any_channel_endpoint_fraction'],
                               clip_stats(aligned_ref[m])['any_channel_endpoint_fraction'])+.003
                       for name,m in audit_masks.items()}
    candidates=[]
    best=None

    def assess(full, item, selected_lut=None):
        nonlocal best
        rgb,compressed=to_rgb(full)
        actual=lab(cv2.GaussianBlur(cv2.resize(rgb,size,interpolation=cv2.INTER_AREA),(0,0),1.5))
        metric=errors(ra[hold],actual.reshape(-1,3)[hold])
        clipping=clip_stats(rgb)
        actual_full=lab(rgb)
        skin_delta=actual_full[skin]-full_lab[skin]
        skin_change={'median_abs_delta_L':float(np.median(np.abs(skin_delta[:,0]))),
                     'p95_abs_delta_L':float(np.percentile(np.abs(skin_delta[:,0]),95)),
                     'p95_chroma_vector_change':float(np.percentile(np.linalg.norm(skin_delta[:,1:],axis=1),95))} if skin.any() else None
        rejected=[]
        if skin_change and (skin_change['p95_abs_delta_L']>6 or skin_change['p95_chroma_vector_change']>5):
            rejected.append('skin_change_limit')
        for key in ['all_channels_near_black_fraction','all_channels_near_white_fraction']:
            if clipping[key]>original_clips[key]+.001:
                rejected.append('new_neutral_clipping')
        for name,m in audit_masks.items():
            if clip_stats(rgb[m])['any_channel_endpoint_fraction']>audit_clip_limits[name]:
                rejected.append('new_channel_endpoint_'+name)
        item={**item,'validation_on_final_RGB':metric,'heldout':metric,'clip':clipping,
              'gamut_compressed_fraction':compressed,'skin_change_not_accuracy':skin_change,'rejected':rejected}
        candidates.append(item)
        if not rejected and metric['selection_score']<original_score['selection_score']*.99 and (best is None or metric['selection_score']<best[0]):
            best=(metric['selection_score'],rgb,item,selected_lut)

    for color_mode in ['offset_only','shared_scale_offset']:
        chroma=fit_chroma(sa[train,1:],ra[train,1:],color_mode!='offset_only')
        for strength in [.5,.75,1.]:
            full=apply_transform(full_lab,curve,chroma,strength)
            assess(full,{'color_mode':color_mode,'strength':strength,'chroma_scale_offset':chroma.tolist()})
    if scene.get('residual_lut',True):
        # Identity is an internal seed when no simple curve passes all gates.
        chosen=best[2] if best is not None else {'color_mode':'identity_seed','strength':0.,'chroma_scale_offset':[1.,0.,0.]}
        chroma=np.array(chosen['chroma_scale_offset']); strength=chosen['strength']
        anchors=[]
        for box in scene.get('skin_boxes',[]):
            m=boxes_mask(src.shape,[box],inside=True)
            values=full_lab[m]
            take=(values[:,0]>15)&(values[:,0]<90)&(values[:,1]>2)&(values[:,2]>4)
            if take.sum()>20:
                anchors.append(np.median(values[take],axis=0))
        grid,correction=fit_residual_lut(sa[train],ra[train],curve,chroma,strength,anchors)
        for residual_strength in [.5,.75,1.]:
            lut=complete_lut(grid,correction,curve,chroma,strength,residual_strength)
            assess(lut_lookup(full_lab,lut),{'color_mode':chosen['color_mode']+'+regularized_3D_LUT',
                   'strength':strength,'residual_strength':residual_strength,'chroma_scale_offset':chroma.tolist(),
                   'skin_color_anchors':np.array(anchors).tolist(),'lut_grid_size':33},lut)
    path=outdir/scene['output']
    if path.exists() or path.resolve() in [Path(p).resolve() for p in before_hash]:
        raise ValueError('Refusing to overwrite output or source.')
    if best is None:
        rgb=src.copy(); selection={'status':'no_candidate_improved_safely','heldout':original_score}
    else:
        _,rgb,selection,chosen_lut=best
        selection={'status':'selected_by_heldout_metrics_pending_visual_review',**selection}
        if chosen_lut is not None:
            lut_path=outdir/(scene['id']+'_selected_lut.npz')
            np.savez_compressed(lut_path,Lab_lut=chosen_lut,domain_min=[0,-128,-128],domain_max=[100,128,128])
            selection['lut_file']=str(lut_path.resolve())
            selection['lut_sha256']=sha256(lut_path)
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(rgb).save(path,icc_profile=icc)
    regions={}
    for name,box in scene.get('audit_regions',{}).items():
        m=boxes_mask(src.shape,[box],inside=True)
        regions[name]={'box':box,'reference':describe(aligned_ref,m),'input':describe(src,m),'corrected':describe(rgb,m)}
    assert all(sha256(p)==checksum for p,checksum in before_hash.items()), 'An input changed during execution.'
    with Image.open(path) as saved:
        assert saved.size==(w,h)
    return {'id':scene['id'],'reference':scene['reference'],'input':scene['input'],'output':str(path.resolve()),
            'input_hashes':before_hash,'output_sha256':sha256(path),'size':[w,h],
            'color_management':{'reference':ref_profile,'input':src_profile,'output':'embedded_sRGB'},
            'geometry_check':geo,'background_coverage_fraction':float(np.mean(valid)),
            'fit_samples':len(train),'validation_samples':len(hold),'validation_used_for_selection_not_independent_test':True,
            'brightness_curve_Lstar':{'input':curve[0].tolist(),'target':curve[1].tolist()},
            'before_heldout':original_score,'candidates':candidates,'selected':selection,'audit_regions':regions}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    outdir=args.config.resolve().parent
    if (outdir/'grading_report.json').exists():
        raise ValueError('Report exists. Use a new versioned output directory.')
    report={'baseline_id':config['baseline_id'],'method':'paired low-frequency background; monotone Lstar curve; bounded chroma scale/offset; optional regularized 33-cube Lab LUT with skin-color anchors; one global color transform, gamut compression at fixed Lstar',
            'limits':'No physical EV, camera white balance or phone-ISP calibration; no local relighting, texture/noise synthesis or geometry edits. Skin ROI change is not identity or skin-color accuracy. Metric selection needs visual review.',
            'candidates_from_original_not_cumulative':True,'scenes':[]}
    for scene in config['scenes']:
        result=process_scene(scene,outdir)
        report['scenes'].append(result)
        print(json.dumps({'id':result['id'],'output':result['output'],'before':result['before_heldout'],
                          'selected':result['selected']},ensure_ascii=False),flush=True)
    (outdir/'grading_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

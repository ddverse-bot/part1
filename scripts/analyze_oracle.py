import sys, json, numpy as np, torch
sys.path.insert(0, "octreg")
from octreg.refine import params_from_matrix
from octreg.common import to_t
from scipy import ndimage
W="work/i46"
A_mri=np.load(f"{W}/mri_affine.npy"); A_oct=np.load(f"{W}/oct150_affine.npy"); oct150=np.load(f"{W}/oct150.npy", mmap_mode="r")
sh=np.array(oct150.shape)
c_o=(A_oct@np.r_[(sh-1)/2.0,1.0])[:3]
T0=np.load("work/runs/crop_mixed_real/T_oct2mri.npy"); B1=np.load("work/runs/validate_real/T_oracle_affine.npy"); A=np.load("work/runs/validate_real/T_auto_vessel.npy")
for name,T in [("T0",T0),("B1 oracle",B1),("A auto",A)]:
    r,t,ls,sh,mir=params_from_matrix(T, c_o)
    print(name, "rotvec deg", np.round(np.degrees(r),2), "centre", np.round(t,2), "scales", np.round(np.exp(ls),3), "shear", np.round(sh,3), "mirror", mir)
# displacement field T0 vs B1 in OCT voxel frame: for OCT voxel p, MRI point B1 p ; map back through T0^-1 -> OCT world ; diff in OCT world coords (axes of OCT array = z,y,x rows of A_oct)
ijk=np.stack(np.meshgrid(*[np.linspace(0,s-1,5) for s in sh],indexing="ij"),-1).reshape(-1,3)
pw=(A_oct@np.c_[ijk,np.ones(len(ijk))].T).T
d=(np.linalg.inv(T0)@B1@pw.T).T[:,:3]-pw[:,:3]      # residual in OCT world (mm)
# express along OCT array axes
R=A_oct[:3,:3]/np.linalg.norm(A_oct[:3,:3],axis=0)
d_ax=d@R   # components along OCT axes z,y,x
print("T0->B1 residual in OCT-axis frame (mm): mean", np.round(d_ax.mean(0),3), "abs mean", np.round(np.abs(d_ax).mean(0),3), "max", np.round(np.abs(d_ax).max(0),3))
print("OCT axis vectors in MRI world:", np.round(R,3).T.tolist(), "voxel size", np.round(np.linalg.norm(A_oct[:3,:3],axis=0),3))
# manual vessel appearance in MRI
mri=np.load(f"{W}/mri.npy", mmap_mode="r"); ves=np.load(f"{W}/mri_vessels.npy", mmap_mode="r"); tissue=np.load(f"{W}/mri_tissue.npy", mmap_mode="r")
lab=np.load(f"{W}/labels4.npy", mmap_mode="r")
idx=np.argwhere(np.asarray(ves)); lo=idx.min(0)-8; hi=idx.max(0)+9
m=np.asarray(mri[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(np.float32); v=np.asarray(ves[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(bool)
t=np.asarray(tissue[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(bool); l=np.asarray(lab[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]])
print("manual vessel voxels", v.sum(), "bbox", lo, hi, "size mm", (hi-lo)*0.15)
loc=ndimage.median_filter(m, size=7)
print("MRI intensity at vessels: mean %.1f (median %.1f); local median at those voxels %.1f; tissue overall mean %.1f"%(m[v].mean(), np.median(m[v]), loc[v].mean(), m[t].mean()))
ratio=m[v]/np.maximum(loc[v],1); print("ratio to local median: median %.2f, p10 %.2f p90 %.2f"%(np.median(ratio),np.percentile(ratio,10),np.percentile(ratio,90)))
print("label at vessel voxels (0 bg,1 wm,2 infra,3 supra):", np.bincount(l[v].astype(int),minlength=4)/v.sum())
print("tissue at vessel voxels:", t[v].mean())
lc,nc=ndimage.label(v); sizes=ndimage.sum(v,lc,index=np.arange(1,nc+1)); print("components", nc, "size percentiles", np.percentile(sizes,[10,50,90,100]))
# radius via EDT inside vessel mask
edt=ndimage.distance_transform_edt(v); print("vessel radius (voxels) at vessel voxels: p50 %.2f p90 %.2f max %.2f"%(np.percentile(edt[v],50),np.percentile(edt[v],90),edt[v].max()))
# vesselness rank test: frangi on region, rank of vessel voxels within tissue and within GM
from octreg.features import frangi_dark_vesselness
vs=frangi_dark_vesselness(to_t(m)[None])[0].cpu().numpy()
for nm,mask in [("tissue",t),("GM(label 2,3)",np.isin(l,(2,3))),("WM",l==1)]:
    vals=vs[mask]; thr=np.percentile(vals,99.7); prec=(v&mask&(vs>thr)).sum()/max((mask&(vs>thr)).sum(),1); rec=(v&mask&(vs>thr)).sum()/max((v&mask).sum(),1)
    print(f"dark-Frangi top0.3% within {nm}: precision {prec:.3f} recall {rec:.3f}; vessel voxels in {nm}: {(v&mask).sum()}")
# bright?
vsb=frangi_dark_vesselness(to_t(-m)[None])[0].cpu().numpy()
for nm,mask in [("tissue",t),("GM(label 2,3)",np.isin(l,(2,3)))]:
    vals=vsb[mask]; thr=np.percentile(vals,99.7); prec=(v&mask&(vsb>thr)).sum()/max((mask&(vsb>thr)).sum(),1); rec=(v&mask&(vsb>thr)).sum()/max((v&mask).sum(),1)
    print(f"BRIGHT-Frangi top0.3% within {nm}: precision {prec:.3f} recall {rec:.3f}")
# simple dark-relative-to-local detector
rel=m/np.maximum(loc,1)
for q in [0.6,0.7,0.8]:
    for nm,mask in [("tissue",t),("GM",np.isin(l,(2,3)))]:
        c=(rel<q)&mask; prec=(v&c).sum()/max(c.sum(),1); rec=(v&c).sum()/max((v&mask).sum(),1)
        print(f"rel<{q} within {nm}: n {c.sum()} precision {prec:.3f} recall {rec:.3f}")

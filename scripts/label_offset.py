import numpy as np, sys, torch
sys.path.insert(0,"octreg")
from octreg.common import to_t, world_bbox_to_voxel
from octreg.features import frangi_dark_vesselness
from scipy import ndimage
W="work/i46"
A_mri=np.load(f"{W}/mri_affine.npy"); mri=np.load(f"{W}/mri.npy",mmap_mode="r"); ves=np.load(f"{W}/mri_vessels.npy",mmap_mode="r"); tissue=np.load(f"{W}/mri_tissue.npy",mmap_mode="r"); lab=np.load(f"{W}/labels4.npy",mmap_mode="r")
A_oct=np.load(f"{W}/oct150_affine.npy"); sh=np.array([50,102,98]); c_o=(A_oct@np.r_[(sh-1)/2,1])[:3]
T0=np.load("work/runs/crop_mixed_real/T_oct2mri.npy"); centre=(T0@np.r_[c_o,1])[:3]
lo,hi=world_bbox_to_voxel(A_mri,mri.shape,centre-14,centre+14)
m=np.asarray(mri[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(np.float32); v=np.asarray(ves[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(bool); t=np.asarray(tissue[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(bool)
print("region",m.shape,"manual vessel voxels",v.sum())
loc=ndimage.median_filter(m,size=9); dark=np.clip(loc-m,0,None)*t
vs=frangi_dark_vesselness(to_t(m)[None])[0].cpu().numpy()*t
def shifted_mean(field, mask, s):
    f=np.roll(field, shift=s, axis=(0,1,2)); return f[mask].mean()
best=[]
for dz in range(-4,5):
    for dy in range(-4,5):
        for dx in range(-4,5):
            best.append((shifted_mean(dark,v,(dz,dy,dx)), shifted_mean(vs,v,(dz,dy,dx)), dz,dy,dx))
b=sorted(best,key=lambda r:-r[0])[:8]
print("top shifts by darkness at label voxels (dark, vesselness, dz,dy,dx):"); [print("  %.2f %.4f  %d %d %d"%r) for r in b]
z=[r for r in best if r[2:]==(0,0,0)][0]; print("zero shift: dark %.2f vess %.4f"%(z[0],z[1]))
b2=sorted(best,key=lambda r:-r[1])[:5]; print("top shifts by vesselness:"); [print("  %.2f %.4f  %d %d %d"%r) for r in b2]
# darkness percentile of label voxels vs tissue voxels
print("darkness at labels p50 %.2f p90 %.2f; tissue p50 %.2f p90 %.2f p99 %.2f"%(np.percentile(dark[v],50),np.percentile(dark[v],90),np.percentile(dark[t],50),np.percentile(dark[t],90),np.percentile(dark[t],99)))
# how much of the strongly dark blobs are near a manual label (within 2 voxels)?
edt=ndimage.distance_transform_edt(~v)
for q in (99,99.5,99.8):
    thr=np.percentile(dark[t],q); c=(dark>thr)&t
    print(f"dark>{q}pct: n {c.sum()} within 2 vox of manual label: {(edt[c]<=2).mean():.2f}, within 4 vox: {(edt[c]<=4).mean():.2f}")
# converse: fraction of manual labels within 2 voxels of a dark>99pct blob
thr=np.percentile(dark[t],99); c=(dark>thr)&t; e2=ndimage.distance_transform_edt(~c); print("manual labels within 2 vox of dark>99pct: %.2f, within 4: %.2f"%((e2[v]<=2).mean(),(e2[v]<=4).mean()))

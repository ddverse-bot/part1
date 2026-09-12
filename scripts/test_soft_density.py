# quick: soft vesselness density vs binary, 24 um, in the vascular refinement
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"octreg")
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, avg_pool_iso, grid_points
from octreg.features import build_features
from octreg.evaluate import transform_diff, gm_wm_overlap
from octreg.vascular import mri_dark_channel, frangi_chunked, density_on_grid, vascular_refine
W="work/i46"; P="work/parser_real"
A_mri=np.load(f"{W}/mri_affine.npy"); mri=np.load(f"{W}/mri.npy",mmap_mode="r"); labels4=np.load(f"{W}/labels4.npy",mmap_mode="r"); tissue=np.load(f"{W}/mri_tissue.npy",mmap_mode="r")
oct150=np.load(f"{W}/oct150.npy"); oct_mask=np.load(f"{W}/oct150_mask.npy"); A_oct=np.load(f"{W}/oct150_affine.npy"); oct_prob=np.load(f"{P}/oct150_prob.npy").astype(np.float32)
T0=np.load("work/runs/crop_final/T_oct2mri_structural.npy"); Tref=np.load("work/runs/validate_real/T_oracle_affine.npy")
sh=np.array(oct150.shape); c_o=(A_oct@np.r_[(sh-1)/2,1])[:3]; centre=(T0@np.r_[c_o,1])[:3]
lo,hi=world_bbox_to_voxel(A_mri,mri.shape,centre-22,centre+22); A_reg=A_mri.copy(); A_reg[:3,3]=(A_mri@np.r_[lo,1])[:3]
mri_reg=np.asarray(mri[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(np.float32); tis=np.asarray(tissue[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(bool)
def reg(n): return np.asarray(np.load(f"{P}/mri_prob_{n}.npy",mmap_mode="r")[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(np.float32)
FM2=torch.stack([to_t(reg("wm")),to_t(reg("gm"))],0)
pred=oct_prob.argmax(0); wm_bright=bool(oct150[(pred==1)&oct_mask].mean()>oct150[(pred>=2)&oct_mask].mean())
FO2=build_features("otsu",to_t(oct150)[None],to_t(oct_mask)[None].float(),wm_bright=wm_bright); MO=to_t(oct_mask)[None].float()
oct_class=np.where(oct_mask,np.where(FO2[0].cpu().numpy()>0.5,1,2),0)
mri_v=mri_dark_channel(mri_reg,tis)
mri_ves=np.load(f"{W}/mri_vessels.npy",mmap_mode="r"); man=np.asarray(mri_ves[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]).astype(bool)
man_w=to_t((A_reg@np.c_[np.argwhere(man),np.ones(int(man.sum()))].T).T[:,:3])
d0um=np.load(f"{W}/oct_ves_dist48_z0.npy"); A0=np.load(f"{W}/oct_ves_dist48_z0_affine.npy"); EDT_O=to_t(d0um/1000.)[None]; A_O48=to_t(A0); sh48=to_t(np.array(d0um.shape)-1)
R=A_oct[:3,:3]/np.linalg.norm(A_oct[:3,:3],axis=0)
def score(T):
    with torch.no_grad():
        p=apply_affine(torch.linalg.inv(to_t(T)),man_w); v=apply_affine(torch.linalg.inv(A_O48),p); ins=((v>=0)&(v<=sh48)).all(1); d=sample_at_world(EDT_O,A_O48,p)[0][ins]*1000
    return float(d.median()), float((d<=150).float().mean())
def rep(name,T,info):
    vm,vf=score(T); gw=gm_wm_overlap(T,oct_class,oct_mask,A_oct,np.asarray(labels4),A_mri); st=[round(float(np.linalg.norm(T[:3,:3]@R[:,q])),3) for q in range(3)]
    print(f"{name:40s} vessels {vm:4.0f}/{vf:.2f} | Dice {gw['dice_WM']:.3f}/{gw['dice_GM']:.3f} | stretch {st} | vs oracle {transform_diff(T,Tref,c_o)['corner_mean_mm']:.2f} | ncc {np.round(info['ncc_channels'],3).tolist()}", flush=True)
oct24=np.load(f"{W}/oct24.npy").astype(np.float32); m24=np.load(f"{W}/oct24_mask.npy"); A24=np.load(f"{W}/oct24_affine.npy")
x=to_t(oct24); t=to_t(m24,dtype=torch.bool)
v=frangi_chunked(x,(1.0,1.5,2.2,3.0,4.4))
OM=to_t(oct_mask,dtype=torch.bool); pts=grid_points(to_t(A_oct),oct150.shape).reshape(-1,3)
def soft_density(vv, clipq):
    vv=vv*t.float(); k=vv[t].flatten().kthvalue(int(clipq*int(t.sum()))).values; s=(vv/k).clamp(0,1)
    dens,A_d=avg_pool_iso(s[None],A24,6); d=sample_at_world(dens,to_t(A_d),pts)[0].reshape(oct150.shape)
    kq=d[OM].flatten().kthvalue(int(0.995*int(OM.sum()))).values.clamp(min=1e-6); return (d/kq).clamp(0,1)*OM.float()
variants={}
thr=v[t].flatten().kthvalue(int(0.99*int(t.sum()))).values; variants["binary top1% (default)"]=density_on_grid((v>thr)&t,A24,A_oct,oct150.shape,oct_mask,pool=6)
variants["soft clip p99.5"]=soft_density(v,0.995); variants["soft clip p99"]=soft_density(v,0.99); variants["soft clip p98"]=soft_density(v,0.98)
# log density of binary
variants["binary top2%"]=density_on_grid((v>v[t].flatten().kthvalue(int(0.98*int(t.sum()))).values)&t,A24,A_oct,oct150.shape,oct_mask,pool=6)
for name,dv in variants.items():
    T,info=vascular_refine(T0,FM2,A_reg,FO2,A_oct,MO,mri_v,dv,w=1.0,reg=0.5,clamp=0.3); rep(name,T,info)
    if name.startswith("binary top1"):
        for w_ in (0.5,2.0):
            T,info=vascular_refine(T0,FM2,A_reg,FO2,A_oct,MO,mri_v,dv,w=w_,reg=0.5,clamp=0.3); rep(f"  binary top1% w={w_}",T,info)
        # sqrt density (compress)
        T,info=vascular_refine(T0,FM2,A_reg,FO2,A_oct,MO,mri_v,dv.sqrt(),w=1.0,reg=0.5,clamp=0.3); rep("  binary top1% sqrt(density)",T,info)

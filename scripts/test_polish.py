import sys, numpy as np, torch
sys.path.insert(0,"octreg")
from octreg.common import to_t, avg_pool_iso
from octreg.refine import Refiner
from octreg.evaluate import transform_diff
W="work/xiangrui_I58bs"
mri=np.load(f"{W}/mri.npy",mmap_mode="r"); tis=np.load(f"{W}/mri_tissue.npy",mmap_mode="r")
o=np.load(f"{W}/oct150.npy"); om=np.load(f"{W}/oct150_mask.npy"); A_o=np.load(f"{W}/oct150_affine.npy"); A_m=np.load(f"{W}/mri_affine.npy")
T0=np.load("work/runs/xiangrui_I58bs_novasc/T_oct2mri.npy")
c_o=(A_o@np.r_[(np.array(o.shape)-1)/2,1])[:3]
m=np.asarray(mri).astype(np.float32); t=np.asarray(tis).astype(bool)
# z-score intensity inside tissue; zero outside
def norm(v,mask):
    x=v.copy(); mu=v[mask].mean(); sd=v[mask].std()+1e-6
    x=(x-mu)/sd; x[~mask]=0; return x
FM=to_t(norm(m,t))[None]; FO=to_t(norm(o,om.astype(bool)))[None]; MO=to_t(om)[None].float()
for lvl,f_m,f_o,iters in ((0.32,4,2,200),(0.16,2,1,250)):
    a1,A1=avg_pool_iso(FM,A_m,f_m); a2,A2=avg_pool_iso(FO,A_o,f_o); a3,_=avg_pool_iso(MO,A_o,f_o)
    ref=Refiner(a1,A1,a2,A2,a3)
    for dof in ("rigid","affine"):
        T0,loss=ref.refine(T0,dof=dof,iters=iters,ls_clamp=0.2,sh_clamp=0.2,reg=1.0)
    print(f"level {lvl}mm: NCC {1-loss:.3f}")
Ts=np.load("work/runs/xiangrui_I58bs_novasc/T_oct2mri.npy")
print("moved vs structural:", {k:round(v,2) for k,v in transform_diff(T0,Ts,c_o).items()})
np.save("work/runs/xiangrui_I58bs_novasc/T_polish.npy",T0)

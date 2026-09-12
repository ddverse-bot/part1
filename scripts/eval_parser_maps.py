import numpy as np, sys
pd=sys.argv[1]
lab=np.load("work/i46/labels4.npy",mmap_mode="r")[::3,::3,::3]; tis=np.load("work/i46/mri_tissue.npy",mmap_mode="r")[::3,::3,::3].astype(bool)
mri=np.load("work/i46/mri.npy",mmap_mode="r")[::3,::3,::3].astype(np.float32); air=(~tis)&(mri<np.percentile(mri[tis],1)); labelled=lab>0
wm=np.load(f"work/{pd}/mri_prob_wm.npy",mmap_mode="r")[::3,::3,::3].astype(np.float32); gm=np.load(f"work/{pd}/mri_prob_gm.npy",mmap_mode="r")[::3,::3,::3].astype(np.float32); t=np.load(f"work/{pd}/mri_prob_tissue.npy",mmap_mode="r")[::3,::3,::3].astype(np.float32)
pred=np.where((1-t)>np.maximum(wm,gm),0,np.where(wm>gm,1,2)); out={}
for c,name,Lm in ((1,"WM",lab==1),(2,"GM",np.isin(lab,(2,3)))):
    P=pred==c; out[name+"_labelled"]=2*(Lm&P&labelled).sum()/(Lm.sum()+(P&labelled).sum())
out["tissue_vs_mask"]=2*((pred>0)&tis).sum()/((pred>0).sum()+tis.sum()); out["air_as_tissue"]=(t[air]>0.5).mean(); out["tissue_as_tissue"]=(t[tis]>0.5).mean()
print(pd,{k:round(float(v),3) for k,v in out.items()})

import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W="work/xiangrui_I58bs"
mri=np.load(f"{W}/mri.npy",mmap_mode="r"); tis=np.load(f"{W}/mri_tissue.npy",mmap_mode="r")
o=np.load(f"{W}/oct150.npy"); om=np.load(f"{W}/oct150_mask.npy")
fig,ax=plt.subplots(3,4,figsize=(17,11))
for r,frac in enumerate((0.3,0.5,0.7)):
    zm=int(mri.shape[0]*frac); zo=int(o.shape[0]*frac)
    m=np.asarray(mri[zm]).astype(np.float32)
    ax[r,0].imshow(np.clip(m/np.percentile(m,99.5),0,1),cmap="gray"); ax[r,0].set_title(f"MRI slice i={zm}")
    ax[r,1].imshow(np.asarray(tis[zm]),cmap="gray"); ax[r,1].set_title("MRI tissue mask")
    ax[r,2].imshow(np.clip(o[zo]/np.percentile(o[om],99),0,1),cmap="gray"); ax[r,2].set_title(f"OCT150 slice z={zo}")
    ax[r,3].imshow(om[zo],cmap="gray"); ax[r,3].set_title("OCT tissue mask")
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.savefig("work/xiangrui_I58bs/diag.png",dpi=90); print("saved diag.png")
sub=np.asarray(mri[::4,::4,::4]).astype(np.float32); pos=sub[sub>0]
h,e=np.histogram(pos,bins=80,range=(0,np.percentile(pos,99.5)))
print("MRI hist:", " ".join(f"{0.5*(e[i]+e[i+1]):.2f}:{h[i]}" for i in range(0,80,4)))
print("MRI frac==0:", float((sub==0).mean()).__round__(3), "tissue frac", float(np.asarray(tis[::4,::4,::4]).mean()).__round__(3))
so=o[om]; print("OCT stats p1/50/99", np.percentile(so,[1,50,99]).round(1), "oct mask frac", float(om.mean()).__round__(3))

import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter, binary_closing, binary_fill_holes, label
from skimage.filters import threshold_otsu
def texture_mask(o, win=7, sig=2.0):
    x=o.astype(np.float32)
    m=uniform_filter(x,win); v=uniform_filter(x*x,win)-m*m
    t=gaussian_filter(np.sqrt(np.maximum(v,0)),sig)
    pos=t[(t>0)&(x>0)]; thr=threshold_otsu(pos[::max(1,pos.size//2_000_000)])
    mask=t>thr
    mask=binary_closing(mask,iterations=2); mask=binary_fill_holes(mask)
    lab,n=label(mask)
    if n>1:
        sizes=np.bincount(lab.ravel())[1:]; mask=lab==(int(np.argmax(sizes))+1)
    return mask, t, float(thr)
fig,ax=plt.subplots(2,4,figsize=(18,9))
for r,(W,name) in enumerate([("work/xiangrui_I58bs","xiangrui I58 brainstem"),("work/I46","I46 Broca")]):
    o=np.load(f"{W}/oct150.npy"); om=np.load(f"{W}/oct150_mask.npy")
    mask,t,thr=texture_mask(o)
    z=o.shape[0]//2
    sl=o[z]; lo,hi=np.percentile(sl[sl>0],[2,98]) if (sl>0).any() else (0,1)
    ax[r,0].imshow(np.clip((sl-lo)/(hi-lo+1e-6),0,1),cmap="gray"); ax[r,0].set_title(f"{name}: oct150 z={z}")
    ax[r,1].imshow(np.clip(t[z]/np.percentile(t[t>0],99),0,1),cmap="gray"); ax[r,1].set_title("texture score")
    ax[r,2].imshow(mask[z],cmap="gray"); ax[r,2].set_title(f"texture mask (frac {mask.mean():.2f})")
    ax[r,3].imshow(om[z],cmap="gray"); ax[r,3].set_title(f"old intensity mask (frac {om.mean():.2f})")
    print(name,"texture frac %.3f, old frac %.3f, agree %.3f"%(mask.mean(), om.mean(), (mask==om).mean()))
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.savefig("work/texture_mask_test.png",dpi=90); print("saved")

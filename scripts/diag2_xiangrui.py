import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W="work/xiangrui_I58bs"
o=np.load(f"{W}/oct150.npy"); om=np.load(f"{W}/oct150_mask.npy")
fig,ax=plt.subplots(2,4,figsize=(18,9))
z=int(o.shape[0]*0.5)
sl=o[z]; lo,hi=np.percentile(sl[sl>0],[2,98])
ax[0,0].imshow(np.clip((sl-lo)/(hi-lo),0,1),cmap="gray"); ax[0,0].set_title(f"oct150 z={z} p2-p98 window")
lo2,hi2=np.percentile(sl[sl>0],[40,99.5])
ax[0,1].imshow(np.clip((sl-lo2)/(hi2-lo2),0,1),cmap="gray"); ax[0,1].set_title("p40-p99.5 window (bright detail)")
from scipy.ndimage import uniform_filter
loc=uniform_filter(sl.astype(np.float32),7); var=uniform_filter(sl.astype(np.float32)**2,7)-loc**2
ax[0,2].imshow(np.clip(np.sqrt(np.maximum(var,0))/np.percentile(np.sqrt(np.maximum(var,0)),99),0,1),cmap="gray"); ax[0,2].set_title("local std (texture)")
h,e=np.histogram(sl[sl>0],bins=100); ax[0,3].plot(0.5*(e[1:]+e[:-1]),h); ax[0,3].set_title("slice histogram"); ax[0,3].axis("on")
y=int(o.shape[1]*0.5)
sly=o[:,y,:]; lo,hi=np.percentile(sly[sly>0],[2,98])
ax[1,0].imshow(np.clip((sly-lo)/(hi-lo),0,1),cmap="gray",aspect="auto"); ax[1,0].set_title(f"oct150 y={y} (z-x plane)")
ax[1,1].imshow(om[:,y,:],cmap="gray",aspect="auto"); ax[1,1].set_title("mask y-slice")
x=int(o.shape[2]*0.5)
slx=o[:,:,x]; lo,hi=np.percentile(slx[slx>0],[2,98])
ax[1,2].imshow(np.clip((slx-lo)/(hi-lo),0,1),cmap="gray",aspect="auto"); ax[1,2].set_title(f"oct150 x={x} (z-y plane)")
ax[1,3].imshow(om[:,:,x],cmap="gray",aspect="auto"); ax[1,3].set_title("mask x-slice")
for i,a_ in enumerate(ax.ravel()):
    if i!=3: a_.axis("off")
plt.tight_layout(); plt.savefig(f"{W}/diag2.png",dpi=90); print("saved")

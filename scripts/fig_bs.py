import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "octreg")
from octreg.common import to_t, apply_affine, sample_at_world, grid_points
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W, R = Path("work/xiangrui_I58bs"), Path("work/runs/xiangrui_I58bs_novasc")
mri = np.load(W / "mri.npy", mmap_mode="r"); A_m = np.load(W / "mri_affine.npy")
o = np.load(W / "oct150.npy"); om = np.load(W / "oct150_mask.npy"); A_o = np.load(W / "oct150_affine.npy")
T = np.load(R / "T_oct2mri.npy"); Tinv = np.linalg.inv(T)
D = o.shape[0]; z = D // 2
c_o = (A_o @ np.r_[(np.array(o.shape) - 1) / 2.0, 1.0])[:3]; bc = (T @ np.r_[c_o, 1.0])[:3]
i0 = int(round(np.linalg.solve(A_m[:3, :3], bc - A_m[:3, 3])[0])); i0 = max(0, min(mri.shape[0] - 1, i0))
sl = np.asarray(mri[i0]).astype(np.float32)
jj, kk = np.meshgrid(np.arange(mri.shape[1]), np.arange(mri.shape[2]), indexing="ij")
pts = (A_m @ np.c_[np.full(jj.size, i0), jj.ravel(), kk.ravel(), np.ones(jj.size)].T).T[:, :3]
with torch.no_grad():
    mo = sample_at_world(to_t(om.astype(np.float32))[None], to_t(A_o), apply_affine(to_t(Tinv), to_t(pts)))[0].reshape(jj.shape).cpu().numpy()
pts_o = grid_points(to_t(A_o), o.shape).reshape(-1, 3)
with torch.no_grad():
    m_in_o = sample_at_world(to_t(np.asarray(mri).astype(np.float32))[None], to_t(A_m), apply_affine(to_t(T), pts_o))[0].reshape(o.shape).cpu().numpy()
vo = np.percentile(o[om], 99); ml, mh = np.percentile(m_in_o[om], [1, 99]); msl_hi = np.percentile(sl[sl > 0], 99.5)
# checkerboard overlay: alternating OCT / registered-MRI tiles show boundary continuity directly
on = np.clip(o[z] / vo, 0, 1); mn = np.clip((m_in_o[z] - ml) / (mh - ml + 1e-6), 0, 1)
t = 32; yy, xx = np.indices(on.shape); board = ((yy // t + xx // t) % 2).astype(bool)
chk = np.where(board, on, mn)
fig, ax = plt.subplots(1, 4, figsize=(23, 6.2))
ax[0].imshow(np.clip(sl / msl_hi, 0, 1), cmap="gray"); ax[0].contour(mo > 0.5, levels=[0.5], colors="red", linewidths=1.4)
ax[0].set_title("MRI (cropped scan) with the located OCT block (red)", fontsize=13)
ax[1].imshow(on, cmap="gray"); ax[1].set_title("OCT, mid-depth slice (raw)", fontsize=13)
ax[2].imshow(mn, cmap="gray"); ax[2].set_title("MRI resampled to the same plane (registered)", fontsize=13)
ax[3].imshow(chk, cmap="gray"); ax[3].set_title("Checkerboard of the two (alignment check)", fontsize=13)
for a_ in ax: a_.axis("off")
plt.tight_layout(); plt.savefig("octreg/figures/fig_I58_brainstem.png", dpi=105, bbox_inches="tight"); plt.close()
print("saved brainstem figure")

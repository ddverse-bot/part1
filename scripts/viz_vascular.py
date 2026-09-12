#!/usr/bin/env python3
"""Figure: does the vascular stage put the MRI dark vessels onto the OCT vessels?  For a few slices through the
block (OCT space, 0.15 mm): OCT with its automatic vessel density; MRI resampled with the structural transform and with
the vascular transform, each with the OCT vessel density contours (green) and the manual MRI vessels (yellow) overlaid.
Rows: 3 slices along the depth axis (top / middle / bottom of the block) — the depth-axis stretch is largest at the ends."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, grid_points, DEVICE
from octreg.vascular import density_on_grid, oct_vessel_mask, mri_dark_channel

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--T-struct", type=Path, required=True); ap.add_argument("--T-vasc", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--T-oracle", type=Path, default=None)
a = ap.parse_args()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r"); mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
Ts = {"structural": np.load(a.T_struct), "vascular": np.load(a.T_vasc)}
if a.T_oracle: Ts["oracle (manual)"] = np.load(a.T_oracle)
D, H, W = oct150.shape
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; centre = (Ts["structural"] @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); ves_reg = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
# OCT vessel density (own Frangi at 24 um, as in the pipeline)
oct24 = np.load(a.work / "oct24.npy").astype(np.float32); m24 = np.load(a.work / "oct24_mask.npy"); A24 = np.load(a.work / "oct24_affine.npy")
dens = density_on_grid(oct_vessel_mask(oct24, m24, q=99.0, sigmas=(1.0, 1.5, 2.2, 3.0, 4.4)), A24, A_oct, oct150.shape, oct_mask, pool=6).cpu().numpy(); del oct24
pts = grid_points(to_t(A_oct), oct150.shape).reshape(-1, 3)
def resample(vol, T, mode="bilinear"):
    return sample_at_world(to_t(vol)[None], to_t(A_reg), apply_affine(to_t(T), pts), mode=mode)[0].reshape(D, H, W).cpu().numpy()
M = {k: resample(mri_reg, T) for k, T in Ts.items()}
V = {k: resample(ves_reg, T, mode="nearest") for k, T in Ts.items()}
vo = np.percentile(oct150[oct_mask], 99); ml, mh = np.percentile(M["structural"][oct_mask], [1, 99])
slices = [int(D * 0.15), D // 2, int(D * 0.85)]
ncol = 1 + len(Ts)
fig, ax = plt.subplots(len(slices), ncol, figsize=(4.2 * ncol, 4.2 * len(slices)))
for r, z in enumerate(slices):
    o = np.clip(oct150[z] / vo, 0, 1)
    ax[r, 0].imshow(o, cmap="gray"); ax[r, 0].contour(dens[z] > 0.35, levels=[0.5], colors="lime", linewidths=0.6)
    ax[r, 0].set_title(f"OCT depth slice {z}/{D} + own vessel density (green)")
    for c, (k, T) in enumerate(Ts.items(), start=1):
        m = np.clip((M[k][z] - ml) / (mh - ml + 1e-6), 0, 1)
        ax[r, c].imshow(m, cmap="gray"); ax[r, c].contour(dens[z] > 0.35, levels=[0.5], colors="lime", linewidths=0.6)
        yy, xx = np.nonzero(V[k][z] > 0.5); ax[r, c].scatter(xx, yy, s=4, c="yellow", marker="s", linewidths=0)
        ax[r, c].contour(oct_mask[z], levels=[0.5], colors="w", linewidths=0.4)
        ax[r, c].set_title(f"MRI via {k} T + OCT vessels (green) + manual MRI vessels (yellow)", fontsize=9)
for x in ax.ravel(): x.axis("off")
plt.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(a.out, dpi=110); plt.close()
print("saved", a.out)

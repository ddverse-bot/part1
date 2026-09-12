#!/usr/bin/env python3
"""Look at the data: MRI orthogonal slices (40x40 mm) through the vessel-label centroid with vessel labels and
GM/WM contours, and the OCT block orthogonal mid-slices at the same physical scale."""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--centre", type=str, default=None, help="world mm x,y,z (default: vessel-label centroid)"); ap.add_argument("--half-mm", type=float, default=20)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
A = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r"); lab = np.load(a.work / "labels4.npy", mmap_mode="r"); ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r")
inv = np.linalg.inv(A)
if a.centre: c = np.array([float(x) for x in a.centre.split(",")])
else:
    ijk = np.argwhere(np.asarray(ves)); c = (A @ np.r_[ijk.mean(0), 1.0])[:3]
print("centre (world mm)", np.round(c, 1).tolist())
# axis-aligned world grid sampling via nearest voxel (MRI axes are permuted world axes, so this is exact)
def slab(axis, coord):
    # returns 2D image in the plane where world[axis]=coord, over the other two axes in [-half, half]
    n = int(2 * a.half_mm / 0.15)
    axes = [i for i in range(3) if i != axis]
    u = np.linspace(-a.half_mm, a.half_mm, n); v = np.linspace(-a.half_mm, a.half_mm, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.zeros((n, n, 3)); P[..., axis] = coord; P[..., axes[0]] = c[axes[0]] + U; P[..., axes[1]] = c[axes[1]] + V
    ijk = np.rint((inv @ np.c_[P.reshape(-1, 3), np.ones(n * n)].T).T[:, :3]).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(mri.shape)), 1)
    def grab(vol, fill=0):
        out = np.full(n * n, fill, dtype=np.float32); out[ok] = np.asarray(vol)[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]; return out.reshape(n, n)
    return grab(mri), grab(lab), grab(ves), axes
fig, ax = plt.subplots(2, 3, figsize=(16, 11))
names = "xyz"
for k, axis in enumerate(range(3)):
    m, l, v, axes = slab(axis, c[axis])
    ax[0, k].imshow(m.T, cmap="gray", vmin=10, vmax=55, origin="lower", extent=[-a.half_mm, a.half_mm, -a.half_mm, a.half_mm])
    ax[0, k].contour(np.linspace(-a.half_mm, a.half_mm, m.shape[0]), np.linspace(-a.half_mm, a.half_mm, m.shape[1]), (l == 1).T, levels=[0.5], colors="r", linewidths=0.6)
    ax[0, k].contour(np.linspace(-a.half_mm, a.half_mm, m.shape[0]), np.linspace(-a.half_mm, a.half_mm, m.shape[1]), np.isin(l, (2, 3)).T, levels=[0.5], colors="c", linewidths=0.6)
    yy, xx = np.nonzero(v.T)
    ax[0, k].scatter(np.linspace(-a.half_mm, a.half_mm, m.shape[0])[xx], np.linspace(-a.half_mm, a.half_mm, m.shape[1])[yy], s=2, c="yellow")
    ax[0, k].set_title(f"MRI plane {names[axis]}={c[axis]:.1f} mm  (h: {names[axes[0]]}, v: {names[axes[1]]}); vessels yellow, WM red, GM cyan")
oct = np.load(a.work / "oct150.npy"); mask = np.load(a.work / "oct150_mask.npy"); D, H, W = oct.shape
vo = np.percentile(oct[mask], 99)
for k, (sl, ttl) in enumerate([((D // 2, slice(None), slice(None)), f"OCT z-mid ({H*0.15:.1f} x {W*0.15:.1f} mm)"), ((slice(None), H // 2, slice(None)), f"OCT y-mid ({D*0.15:.1f} x {W*0.15:.1f} mm)"), ((slice(None), slice(None), W // 2), f"OCT x-mid ({D*0.15:.1f} x {H*0.15:.1f} mm)")]):
    img = oct[sl]; ax[1, k].imshow(img, cmap="gray", vmin=0, vmax=vo, aspect="equal"); ax[1, k].set_title(ttl)
for x in ax.ravel(): x.axis("off")
plt.tight_layout(); plt.savefig(a.out / "region_vs_oct.png", dpi=90); plt.close(); print("wrote", a.out / "region_vs_oct.png")

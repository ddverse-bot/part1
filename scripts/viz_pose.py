#!/usr/bin/env python3
"""Visual check of a candidate pose: OCT slices with MRI vessel labels (mapped into OCT space) and OCT vessel
segmentation overlaid, MRI resampled into OCT space (checkerboard), MRI GM/WM label contours on OCT."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--T", type=str, required=True, help="path to .npy or json:path:rank"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--title", default="")
a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
if a.T.startswith("json:"):
    _, p, rank = a.T.split(":"); T = np.array(json.load(open(p))["rows"][int(rank)]["T"])
else: T = np.load(a.T)
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r"); lab = np.load(a.work / "labels4.npy", mmap_mode="r"); ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
D, H, W = oct150.shape
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; centre = (T @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 16, centre + 16)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); lab_reg = np.asarray(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
# MRI vessel points into OCT voxel coords
ijk = np.argwhere(np.asarray(ves)); pw = (A_mri @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
po = (np.linalg.inv(T) @ np.c_[pw, np.ones(len(pw))].T).T[:, :3]; vo = (np.linalg.inv(A_oct) @ np.c_[po, np.ones(len(po))].T).T[:, :3]
inside = np.all((vo >= 0) & (vo < np.array(oct150.shape)), 1); vo = vo[inside]
# OCT vessel seg on the 0.15 grid (from 48um dist map == 0)
d0 = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
vv = np.argwhere(d0 <= 0); vw = (A0 @ np.c_[vv, np.ones(len(vv))].T).T[:, :3]; vvo = (np.linalg.inv(A_oct) @ np.c_[vw, np.ones(len(vw))].T).T[:, :3]
segvol = np.zeros(oct150.shape, np.float32); ii = np.rint(vvo).astype(int); ok = np.all((ii >= 0) & (ii < np.array(oct150.shape)), 1); np.add.at(segvol, (ii[ok, 0], ii[ok, 1], ii[ok, 2]), 1)
# MRI resampled into OCT space + labels
Ai = to_t(A_oct); Tt = to_t(T)
grid = torch.stack(torch.meshgrid(torch.arange(D, device="cuda", dtype=torch.float32), torch.arange(H, device="cuda", dtype=torch.float32), torch.arange(W, device="cuda", dtype=torch.float32), indexing="ij"), -1).reshape(-1, 3)
pm = apply_affine(Tt, apply_affine(Ai, grid))
mri_in = sample_at_world(to_t(mri_reg)[None], to_t(A_reg), pm)[0].reshape(D, H, W).cpu().numpy()
lab_in = sample_at_world(to_t(lab_reg)[None], to_t(A_reg), pm, mode="nearest")[0].reshape(D, H, W).cpu().numpy()
vo_max = np.percentile(oct150[mask], 99); mlo, mhi = np.percentile(mri_in[mask], [1, 99])
fig, ax = plt.subplots(3, 4, figsize=(18, 12))
for r, (axis, idx) in enumerate([(0, D // 2), (1, H // 2), (2, W // 2)]):
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    o = np.clip(oct150[sl] / vo_max, 0, 1); m = np.clip((mri_in[sl] - mlo) / (mhi - mlo + 1e-6), 0, 1)
    ax[r, 0].imshow(o, cmap="gray"); ax[r, 0].set_title(f"OCT axis{axis}={idx}")
    ax[r, 1].imshow(m, cmap="gray"); ax[r, 1].set_title("MRI in OCT space")
    cb = (np.indices(o.shape).sum(0) // 10) % 2; ax[r, 2].imshow(np.where(cb == 0, o, m), cmap="gray"); ax[r, 2].set_title("checkerboard")
    ax[r, 3].imshow(o, cmap="gray")
    L = lab_in[sl]; ax[r, 3].contour(L == 1, levels=[0.5], colors="r", linewidths=0.7); ax[r, 3].contour(np.isin(L, (2, 3)), levels=[0.5], colors="c", linewidths=0.7)
    # vessel points near this slice: MRI labels (yellow) and OCT seg (green)
    sel = np.abs(vo[:, axis] - idx) <= 1.5; pts = vo[sel]; oth = [i for i in range(3) if i != axis]
    ax[r, 3].scatter(pts[:, oth[1]], pts[:, oth[0]], s=6, c="yellow", alpha=0.9)
    sv = segvol[sl] > 0; yy, xx = np.nonzero(sv); ax[r, 3].scatter(xx, yy, s=1, c="lime", alpha=0.5)
    ax[r, 3].set_title("MRI labels: WM red, GM cyan, vessels yellow; OCT vessels green")
for x in ax.ravel(): x.axis("off")
fig.suptitle(a.title + f"  centre {np.round(centre,1).tolist()} scales {np.round(np.linalg.norm(T[:3,:3],axis=0),2).tolist()} det {np.linalg.det(T[:3,:3]):.2f}")
plt.tight_layout(); plt.savefig(a.out, dpi=85); print("wrote", a.out)

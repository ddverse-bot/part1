#!/usr/bin/env python3
"""One figure per registration run: 3 depth slices of the OCT block; left = OCT with our vessel segmentation (green) and the
WM/GM class boundary (cyan); middle = MRI resampled into the OCT frame with the structural transform; right = with the final
(vascular) transform.  Overlays on the MRI panels: OCT vessels (green), manual MRI vessel label (yellow, if present),
OCT class boundary (cyan), block outline (white)."""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, grid_points
from octreg.vascular import density_on_grid
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args(); out = a.out or (a.run / "qc_vessels.png")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r") if (a.work / "mri_vessels.npy").exists() else None
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_class = np.load(a.run / "oct_class150.npy") if (a.run / "oct_class150.npy").exists() else None
Ts = {"structural": np.load(a.run / "T_oct2mri_structural.npy"), "final": np.load(a.run / "T_oct2mri.npy")}
res = json.load(open(a.run / "result.json"))
D, H, W = oct150.shape
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; centre = (Ts["final"] @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
ves_reg = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32) if mri_ves is not None else None
vmask = to_t(np.load(a.work / "octv_vessels.npy"), dtype=torch.bool); A_vv = np.load(a.work / "octv_affine.npy"); sp_v = np.linalg.norm(A_vv[:3, :3], axis=0) * 1000
dens = density_on_grid(vmask, A_vv, A_oct, oct150.shape, oct_mask, pool=int(max(1, round(150.0 / sp_v.mean())))).cpu().numpy(); del vmask
pts = grid_points(to_t(A_oct), oct150.shape).reshape(-1, 3)
def resample(vol, T, mode="bilinear"): return sample_at_world(to_t(vol)[None], to_t(A_reg), apply_affine(to_t(T), pts), mode=mode)[0].reshape(D, H, W).cpu().numpy()
M = {k: resample(mri_reg, T) for k, T in Ts.items()}
V = {k: resample(ves_reg, T, mode="nearest") for k, T in Ts.items()} if ves_reg is not None else None
vo = np.percentile(oct150[oct_mask], 99); ml, mh = np.percentile(M["final"][oct_mask], [1, 99])
slices = [int(D * 0.2), D // 2, int(D * 0.8)]
fig, ax = plt.subplots(len(slices), 3, figsize=(13, 4.3 * len(slices)))
for r, z in enumerate(slices):
    ax[r, 0].imshow(np.clip(oct150[z] / vo, 0, 1), cmap="gray"); ax[r, 0].contour(dens[z] > 0.35, levels=[0.5], colors="lime", linewidths=0.6)
    if oct_class is not None: ax[r, 0].contour(oct_class[z] == 1, levels=[0.5], colors="cyan", linewidths=0.5)
    ax[r, 0].set_title(f"OCT depth slice {z}/{D}: own vessels (green), WM/GM split (cyan)", fontsize=9)
    for c, k in enumerate(Ts, start=1):
        ax[r, c].imshow(np.clip((M[k][z] - ml) / (mh - ml + 1e-6), 0, 1), cmap="gray"); ax[r, c].contour(dens[z] > 0.35, levels=[0.5], colors="lime", linewidths=0.6)
        if oct_class is not None: ax[r, c].contour(oct_class[z] == 1, levels=[0.5], colors="cyan", linewidths=0.5)
        if V is not None: yy, xx = np.nonzero(V[k][z] > 0.5); ax[r, c].scatter(xx, yy, s=4, c="yellow", marker="s", linewidths=0)
        ax[r, c].contour(oct_mask[z], levels=[0.5], colors="w", linewidths=0.4)
        ax[r, c].set_title(f"MRI via {k} transform" + (" (+ manual MRI vessels, yellow)" if V is not None else ""), fontsize=9)
for x in ax.ravel(): x.axis("off")
ft = res.get("final_transform", {}); s = res.get("search", {})
fig.suptitle(f"{a.work.name} [{res['args'].get('features')}]  search top1/top2 {s.get('top1', 0):.3f}/{s.get('top2', 0):.3f}  mirror {ft.get('mirror')}  stretch ijk {ft.get('stretch_ijk')}", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.98)); out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=100); plt.close(); print("saved", out)

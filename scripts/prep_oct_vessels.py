#!/usr/bin/env python3
"""Cache the OCT vessel-density map for the label-free vascular channel, from our own Frangi segmentation of the OCT
at native (12 um) and/or pooled (24 um) resolution: work/oct_vesdens150_{12um,24um}.npy on the oct150 grid (float16).
CPU-resident volumes, GPU per z-chunk (works next to other GPU jobs).
For another dataset: --oct <stack> (Z,Y,X); needs work/oct12_affine.npy, oct150_affine.npy, oct150_mask.npy from prep."""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import oct_slab_normalize, oct_tissue_mask, pool_mean_np, to_t, grid_points, sample_at_world
from octreg.features import frangi_dark_vesselness
ap = argparse.ArgumentParser()
ap.add_argument("--dandi", type=Path, default=None); ap.add_argument("--oct", type=Path, default=None)
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--levels", type=str, default="12,24", help="um levels to segment (12 = native)")
ap.add_argument("--q", type=float, default=99.0); ap.add_argument("--sigmas", type=str, default="1,1.5,2.2,3,4.4")
ap.add_argument("--slab-window", type=int, default=100); ap.add_argument("--tissue-thresh", type=float, default=30.0)
a = ap.parse_args(); t0 = time.time()
src = a.oct or (a.dandi / "sub-I46/ses-OCT/micr/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff")
if str(src).endswith(".npy"): oct12 = np.load(src, mmap_mode="r")
else:
    import tifffile; oct12 = tifffile.imread(str(src))
octn, _ = oct_slab_normalize(np.asarray(oct12), tissue_thresh=a.tissue_thresh, window=a.slab_window); del oct12
octn = octn.astype(np.float32)
print("OCT normalised", octn.shape, f"({time.time()-t0:.0f}s)", flush=True)
A12 = np.load(a.work / "oct12_affine.npy"); A150 = np.load(a.work / "oct150_affine.npy"); m150 = np.load(a.work / "oct150_mask.npy"); sh150 = m150.shape
sig = tuple(float(x) for x in a.sigmas.split(","))
p48 = pool_mean_np(octn, 4); _, thr = oct_tissue_mask(p48, thresh=None, closing_iter=2); del p48
pts150 = grid_points(to_t(A150), sh150).reshape(-1, 3); OM = to_t(m150, dtype=torch.bool)
for lvl in [int(x) for x in a.levels.split(",")]:
    k = lvl // 12
    vol = octn if k == 1 else pool_mean_np(octn, k)
    A = A12.copy(); A[:3, 3] = A12[:3, 3] + A12[:3, :3] @ (np.ones(3) * (k - 1) / 2.0); A[:3, :3] *= k
    D = vol.shape[0]; zc, ov = (12 if k == 1 else 24), 10
    ves = np.zeros(vol.shape, np.float16)
    with torch.no_grad():
        for z0 in range(0, D, zc):
            lo_, hi_ = max(0, z0 - ov), min(D, z0 + zc + ov)
            vg = frangi_dark_vesselness(to_t(vol[lo_:hi_])[None], sigmas_vox=sig)[0]
            ves[z0:min(D, z0 + zc)] = vg[z0 - lo_:z0 - lo_ + min(zc, D - z0)].cpu().numpy().astype(np.float16)
            del vg
    torch.cuda.empty_cache()
    tis = vol > thr
    sub = ves[tis][::max(1, int(tis.sum()) // 20_000_000)].astype(np.float32); thr_v = float(np.percentile(sub, a.q))
    mask = (ves > thr_v) & tis; del ves
    print(f"level {lvl} um: {vol.shape} tissue {tis.mean():.2f} vessel voxels {int(mask.sum())} thr {thr_v:.4f} ({time.time()-t0:.0f}s)", flush=True)
    pool = max(1, 144 // lvl)
    dens = pool_mean_np(mask.astype(np.float32), pool); del mask, tis
    Ad = A.copy(); Ad[:3, 3] = A[:3, 3] + A[:3, :3] @ (np.ones(3) * (pool - 1) / 2.0); Ad[:3, :3] *= pool
    d = sample_at_world(to_t(dens)[None], to_t(Ad), pts150)[0].reshape(sh150)
    kq = d[OM].flatten().kthvalue(int(0.995 * int(OM.sum()))).values.clamp(min=1e-6)
    d = ((d / kq).clamp(0, 1) * OM.float()).cpu().numpy().astype(np.float16)
    np.save(a.work / f"oct_vesdens150_{lvl}um.npy", d)
    if k != 1: del vol
    print(f"  saved oct_vesdens150_{lvl}um.npy (mean in tissue {float(d[m150].mean()):.3f})", flush=True)
print("done", f"({time.time()-t0:.0f}s)")

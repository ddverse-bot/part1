#!/usr/bin/env python3
"""Prepare the Costantini sub-I46 pair for octreg (cached numpy arrays at 0.15 mm).

Outputs in --work:
  mri.npy / mri_affine.npy          whole-hemisphere MRI (float32, native grid 1280x1040x576, 0.15 mm)
  mri_tissue.npy                    brain-tissue mask (Otsu between fluid and tissue) [bool]
  labels4.npy                       whole-hemisphere labels on the MRI grid: 0 bg, 1 WM, 2 infra GM, 3 supra GM [uint8]
  ba_labels.npy                     BA44/45 manual labels (same coding) [uint8]
  mri_vessels.npy                   MRI vessel label (255) [bool]
  oct150.npy / oct150_affine.npy    OCT block: slab-normalized, mean-pooled 12x, resampled to 0.15 mm [float32]
  oct150_mask.npy                   OCT tissue mask [bool]
  oct12_affine.npy                  affine of the full-resolution OCT array (Z,Y,X) [world frame 'SPR', corner origin]
  prep.json                         parameters / stats
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, tifffile, nibabel as nib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import (DEVICE, layout_affine, oct_slab_normalize, oct_tissue_mask, pool_mean_np,
                           resample_to_grid, to_t, iso_grid_affine, write_json)

ap = argparse.ArgumentParser()
ap.add_argument("--dandi", type=Path, required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--oct-layout", default="SPR")
ap.add_argument("--target-mm", type=float, default=0.15)
a = ap.parse_args()
a.work.mkdir(parents=True, exist_ok=True)
info = {}
t0 = time.time()

# ---------------- MRI + labels
mri_img = nib.load(str(a.dandi / "derivatives/EPIC/sub-I46/ses-MRI/anat/sub-I46_ses-MRI_flip-2_VFA.nii.gz"))
mri = np.asanyarray(mri_img.dataobj).astype(np.float32)
A_mri = np.asarray(mri_img.affine)
np.save(a.work / "mri.npy", mri); np.save(a.work / "mri_affine.npy", A_mri)
from skimage.filters import threshold_otsu
sub = mri[::4, ::4, ::4]; sub = sub[sub > 1]
# fluid mode ~14, tissue 28-50 in this EPIC contrast: take the histogram valley between the two modes
h, e = np.histogram(sub, bins=np.arange(0, 60.5, 0.5))
c = 0.5 * (e[1:] + e[:-1]); sel = (c > 15) & (c < 28)
thr = float(c[sel][np.argmin(h[sel])])
tissue = mri > thr
np.save(a.work / "mri_tissue.npy", tissue)
info["mri"] = {"shape": mri.shape, "tissue_thresh": thr, "tissue_fraction": float(tissue.mean())}
print("mri", mri.shape, "tissue thr", round(thr, 1), f"{time.time()-t0:.0f}s", flush=True)

ba = np.asanyarray(nib.load(str(a.dandi / "derivatives/Labels/sub-I46/ses-MRI/anat/sub-I46_ses-MRI_space-EPIC_label-infrasupra_dseg.nii.gz")).dataobj).astype(np.uint8)
np.save(a.work / "ba_labels.npy", ba)
wh_img = nib.load(str(a.dandi / "derivatives/Labels/sub-I46/ses-MRI/anat/sub-I46_ses-MRI_space-EPIC_label-infrasupra_seg-wholehemi_dseg.nii.gz"))
wh = np.asanyarray(wh_img.dataobj)
A_wh = np.asarray(wh_img.affine)
# map MRI voxel (i,j,k) -> WH voxel via affines; verify it is an integer axis permutation
M = np.linalg.inv(A_wh) @ A_mri
assert np.allclose(np.abs(M[:3, :3]).sum(1), 1, atol=1e-3) and np.allclose(np.round(M[:3, 3]), M[:3, 3], atol=1e-2), M
perm = [int(np.argmax(np.abs(M[r, :3]))) for r in range(3)]   # WH axis r comes from MRI axis perm[r]
sign = [int(np.sign(M[r, perm[r]])) for r in range(3)]
off = np.round(M[:3, 3]).astype(int)
lab = np.zeros(mri.shape, dtype=np.uint8)
# build index arrays
idx = [None, None, None]
for r in range(3):
    n = mri.shape[perm[r]]
    ar = sign[r] * np.arange(n) + off[r]
    idx[r] = ar
ii, jj, kk = np.meshgrid(*[np.arange(s) for s in mri.shape], indexing="ij", sparse=True)
mri_idx = [ii, jj, kk]
wh_idx = [idx[r][mri_idx[perm[r]]] for r in range(3)]
valid = np.ones(mri.shape, dtype=bool)
for r in range(3):
    valid &= (wh_idx[r] >= 0) & (wh_idx[r] < wh.shape[r])
lab_vals = wh[np.clip(wh_idx[0], 0, wh.shape[0]-1), np.clip(wh_idx[1], 0, wh.shape[1]-1), np.clip(wh_idx[2], 0, wh.shape[2]-1)]
lab = np.where(valid, lab_vals, 0).astype(np.uint8)
np.save(a.work / "labels4.npy", lab)
u, c = np.unique(lab, return_counts=True)
info["labels4"] = {"map": "0 bg,1 WM,2 infra GM,3 supra GM", "counts": dict(zip(u.tolist(), c.tolist())), "perm": perm, "sign": sign, "off": off.tolist()}
# consistency check with the BA labels (same coding expected): agreement inside BA region
m = ba > 0
info["labels4"]["agreement_with_BA_labels"] = float((lab[m] == ba[m]).mean())
print("labels4 counts", info["labels4"]["counts"], "agree with BA:", round(info["labels4"]["agreement_with_BA_labels"], 3), flush=True)
del wh
ves = np.asanyarray(nib.load(str(a.dandi / "derivatives/Labels/sub-I46/ses-MRI/anat/sub-I46_ses-MRI_space-EPIC_label-vessels_dseg.nii.gz")).dataobj)
np.save(a.work / "mri_vessels.npy", ves == 255)
del ves

# ---------------- OCT
oct12 = tifffile.imread(str(a.dandi / "sub-I46/ses-OCT/micr/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff"))
A12 = layout_affine(oct12.shape, 0.012, a.oct_layout, False)
np.save(a.work / "oct12_affine.npy", A12)
octn, slab = oct_slab_normalize(oct12, tissue_thresh=30.0, window=100)
del oct12
p12 = pool_mean_np(octn, 12)                       # 0.144 mm
A144 = A12.copy(); A144[:3, :3] *= 12; A144[:3, 3] = A12[:3, 3] + A12[:3, :3] @ (np.ones(3) * 5.5)
# resample onto a 0.15 mm grid aligned with the OCT array axes (same layout, exact spacing)
shape150 = tuple(int(np.floor(s * 0.144 / a.target_mm)) for s in p12.shape)
A150 = A144.copy(); A150[:3, :3] = A12[:3, :3] / 0.012 * a.target_mm; A150[:3, 3] = A144[:3, 3]
v = to_t(p12)[None]
o150 = resample_to_grid(v, to_t(A144), to_t(A150), shape150).cpu().numpy()[0]
mask, thr_o = oct_tissue_mask(o150, thresh=None, closing_iter=2)
np.save(a.work / "oct150.npy", o150.astype(np.float32)); np.save(a.work / "oct150_affine.npy", A150); np.save(a.work / "oct150_mask.npy", mask)
info["oct"] = {"shape12": [627, 1271, 1230], "layout": a.oct_layout, "shape150": list(shape150), "tissue_thresh": thr_o,
               "tissue_fraction": float(mask.mean()), "slab_window": 100,
               "per_slice_median_head": [round(x, 1) if x == x else None for x in slab["per_slice_median"][:5]]}
print("oct150", shape150, "tissue frac", round(float(mask.mean()), 3), flush=True)
write_json(info, a.work / "prep.json")
print("done", f"{time.time()-t0:.0f}s")

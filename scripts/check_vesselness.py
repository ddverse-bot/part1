#!/usr/bin/env python3
"""How well does automatic (annotation-free) dark-tube vesselness recover the manual vessel labels?
MRI: Frangi on the crop vs label-vessels (255); OCT: Frangi on oct150 vs ves_seg pooled to 0.15 mm.
Reports precision/recall at a few thresholds + saves overlays."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, tifffile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t
from octreg.features import frangi_dark_vesselness
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--dandi", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
res = {}
# ---- MRI crop around BA44/45
mri = np.load(a.work / "mri.npy", mmap_mode="r"); ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); tis = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
nz = np.argwhere(np.asarray(ves[::4, ::4, ::4])) * 4; lo = np.maximum(nz.min(0) - 8, 0); hi = np.minimum(nz.max(0) + 8, np.array(mri.shape))
sub = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); vsub = np.asarray(ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); tsub = np.asarray(tis[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
print("MRI crop", sub.shape, flush=True)
v = frangi_dark_vesselness(to_t(sub)[None])[0].cpu().numpy() * tsub
from scipy import ndimage
gt = ndimage.binary_dilation(vsub, iterations=1)   # tolerate 1 voxel
for q in (99.0, 99.5, 99.8):
    thr = np.percentile(v[tsub], q); pred = v > thr
    prec = (pred & gt).sum() / max(pred.sum(), 1); rec = (ndimage.binary_dilation(pred, iterations=1) & vsub).sum() / max(vsub.sum(), 1)
    res[f"mri_q{q}"] = {"thr": float(thr), "precision_vs_dilated_labels": float(prec), "recall_of_labels": float(rec), "n_pred": int(pred.sum())}
print("MRI vesselness:", json.dumps(res, indent=0)[:600], flush=True)
z = sub.shape[0] // 2
fig, ax = plt.subplots(1, 3, figsize=(15, 5)); ax[0].imshow(sub[z], cmap="gray", vmin=10, vmax=55); ax[0].set_title("MRI"); ax[1].imshow(v[z], cmap="magma"); ax[1].set_title("vesselness"); ax[2].imshow(sub[z], cmap="gray", vmin=10, vmax=55); ax[2].contour(vsub[z], levels=[0.5], colors="r", linewidths=0.6); ax[2].contour(v[z] > np.percentile(v[tsub], 99.5), levels=[0.5], colors="y", linewidths=0.6); ax[2].set_title("labels red / vesselness>q99.5 yellow")
for x in ax: x.axis("off")
plt.tight_layout(); plt.savefig(a.out / "mri_vesselness.png", dpi=100); plt.close()
# ---- OCT
oct = np.load(a.work / "oct150.npy"); mask = np.load(a.work / "oct150_mask.npy")
vo = frangi_dark_vesselness(to_t(oct)[None])[0].cpu().numpy() * mask
seg = tifffile.imread(str(a.dandi / "derivatives/sub-I46/ses-OCT/vessel/ves_seg.tif")) > 0
# pool ves_seg to oct150 grid approx: 12x pooling (0.144) then treat as 0.15 (approx OK for a QC)
z_, y_, x_ = (s // 12 * 12 for s in seg.shape)
segp = seg[:z_, :y_, :x_].reshape(z_ // 12, 12, y_ // 12, 12, x_ // 12, 12).mean(axis=(1, 3, 5)) > 0.02
segp_full = np.zeros(oct.shape, bool); zz = min(segp.shape[0], oct.shape[0]); yy = min(segp.shape[1], oct.shape[1]); xx = min(segp.shape[2], oct.shape[2]); segp_full[:zz, :yy, :xx] = segp[:zz, :yy, :xx]
for q in (95.0, 98.0, 99.0):
    thr = np.percentile(vo[mask], q); pred = vo > thr
    gt = ndimage.binary_dilation(segp_full, iterations=1)
    prec = (pred & gt).sum() / max(pred.sum(), 1); rec = (ndimage.binary_dilation(pred, iterations=1) & segp_full).sum() / max(segp_full.sum(), 1)
    res[f"oct_q{q}"] = {"thr": float(thr), "precision_vs_dilated_seg": float(prec), "recall_of_seg": float(rec), "n_pred": int(pred.sum()), "n_seg": int(segp_full.sum())}
print("OCT vesselness:", json.dumps({k: v for k, v in res.items() if k.startswith("oct")}, indent=0)[:600], flush=True)
z = oct.shape[0] // 2
fig, ax = plt.subplots(1, 3, figsize=(15, 5)); ax[0].imshow(oct[z], cmap="gray", vmin=0, vmax=np.percentile(oct[mask], 99)); ax[0].set_title("OCT 0.15mm"); ax[1].imshow(vo[z], cmap="magma"); ax[1].set_title("vesselness"); ax[2].imshow(oct[z], cmap="gray", vmin=0, vmax=np.percentile(oct[mask], 99)); ax[2].contour(segp_full[z], levels=[0.5], colors="r", linewidths=0.6); ax[2].contour(vo[z] > np.percentile(vo[mask], 98), levels=[0.5], colors="y", linewidths=0.6); ax[2].set_title("ves_seg red / vesselness>q98 yellow (z-offset 0 assumed)")
for x in ax: x.axis("off")
plt.tight_layout(); plt.savefig(a.out / "oct_vesselness.png", dpi=100); plt.close()
json.dump(res, open(a.out / "vesselness_check.json", "w"), indent=1)

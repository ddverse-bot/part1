#!/usr/bin/env python3
"""Visual QC of the parser: OCT (3 orthogonal mid-slices: image | argmax overlay) and an MRI patch
(image | parser argmax | labels)."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.parser import load_parser, parse_volume

ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
cmap = np.array([[0, 0, 0, 0], [1, 0.2, 0.2, 0.55], [0.2, 0.9, 0.2, 0.55], [0.2, 0.4, 1.0, 0.55]])  # bg, WM red, infra green, supra blue

oct = np.load(a.work / "oct150.npy"); mask = np.load(a.work / "oct150_mask.npy")
prob = np.load(a.parser.parent / "oct150_prob.npy").astype(np.float32) if (a.parser.parent / "oct150_prob.npy").exists() else None
if prob is None:
    net = load_parser(a.parser); prob = parse_volume(net, oct, patch=64, stride=32).float().numpy()
pred = prob.argmax(0)
D, H, W = oct.shape
fig, ax = plt.subplots(3, 3, figsize=(12, 11))
for r, (sl, title) in enumerate([((D // 2, slice(None), slice(None)), "z mid"), ((slice(None), H // 2, slice(None)), "y mid"), ((slice(None), slice(None), W // 2), "x mid")]):
    img = oct[sl]; pr = pred[sl]; m = mask[sl]
    ax[r, 0].imshow(img, cmap="gray", vmin=0, vmax=np.percentile(oct[mask], 99)); ax[r, 0].set_title(f"OCT {title}")
    ax[r, 1].imshow(img, cmap="gray", vmin=0, vmax=np.percentile(oct[mask], 99)); ax[r, 1].imshow(cmap[pr]); ax[r, 1].set_title("parser argmax (WM red, infra green, supra blue)")
    ax[r, 2].imshow(prob[1][sl], cmap="magma", vmin=0, vmax=1); ax[r, 2].set_title("P(WM)")
for x in ax.ravel(): x.axis("off")
plt.tight_layout(); plt.savefig(a.out / "oct_parser.png", dpi=110); plt.close()

# MRI: BA44/45 region centre slice
mri = np.load(a.work / "mri.npy", mmap_mode="r"); lab = np.load(a.work / "labels4.npy", mmap_mode="r"); ba = np.load(a.work / "ba_labels.npy", mmap_mode="r")
nz = np.argwhere(np.asarray(ba[::8, ::8, ::8]) > 0) * 8; c = nz.mean(0).astype(int)
lo = np.maximum(c - 80, 0); hi = lo + 160
sub = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); sl = np.asarray(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
net = load_parser(a.parser); p = parse_volume(net, sub, patch=96, stride=64).float().numpy(); pr = p.argmax(0)
fig, ax = plt.subplots(3, 3, figsize=(12, 11))
for r, s_ in enumerate([(80, slice(None), slice(None)), (slice(None), 80, slice(None)), (slice(None), slice(None), 80)]):
    ax[r, 0].imshow(sub[s_], cmap="gray", vmin=10, vmax=55); ax[r, 0].set_title("MRI")
    ax[r, 1].imshow(sub[s_], cmap="gray", vmin=10, vmax=55); ax[r, 1].imshow(cmap[pr[s_]]); ax[r, 1].set_title("parser argmax")
    ax[r, 2].imshow(sub[s_], cmap="gray", vmin=10, vmax=55); ax[r, 2].imshow(cmap[sl[s_]]); ax[r, 2].set_title("labels (whole-hemi)")
for x in ax.ravel(): x.axis("off")
plt.tight_layout(); plt.savefig(a.out / "mri_parser.png", dpi=110); plt.close()
print("wrote", a.out)

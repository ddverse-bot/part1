#!/usr/bin/env python3
"""Train the contrast-agnostic structural parser on synthetic images from the whole-hemisphere labels
(+ a fraction of real MRI patches).  Usage:
    python train_parser.py --work work/i46 --out work/parser_v1 --iters 4000
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.parser import train_parser, load_parser, parse_volume, PatchSampler
from octreg.synth import standardize

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--patch", type=int, default=96)
ap.add_argument("--p-real", type=float, default=0.3)
ap.add_argument("--p-invert", type=float, default=0.0, help="probability of inverting the polarity of a real-MRI patch (contrast-agnostic)")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude-centre-mm", type=str, default=None, help="world mm x,y,z: hold out a +-half box around it")
ap.add_argument("--exclude-half-mm", type=float, default=25.0)
a = ap.parse_args()

labels = np.load(a.work / "labels4.npy", mmap_mode="r")
mri = np.load(a.work / "mri.npy", mmap_mode="r")
tissue = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
excl = None
if a.exclude_centre_mm:
    from octreg.common import world_bbox_to_voxel
    A_mri = np.load(a.work / "mri_affine.npy"); c = np.array([float(x) for x in a.exclude_centre_mm.split(",")])
    excl = world_bbox_to_voxel(A_mri, labels.shape, c - a.exclude_half_mm, c + a.exclude_half_mm)
    print("holding out MRI voxel box", excl[0].tolist(), excl[1].tolist(), "from training", flush=True)
print("training parser:", dict(iters=a.iters, batch=a.batch, patch=a.patch, p_real=a.p_real, real_only=a.p_real >= 1.0, p_invert=a.p_invert), flush=True)
ck = train_parser(labels, mri, a.out, iters=a.iters, batch=a.batch, patch=a.patch, p_real=a.p_real, lr=a.lr,
                  tissue_mask=tissue, seed=a.seed, exclude_ijk=excl, p_invert=a.p_invert)

# quick validation: Dice on 16 real MRI patches (labelled voxels only) and OCT class fractions
net = load_parser(ck)
sampler = PatchSampler(np.asarray(labels), np.asarray(mri), patch=a.patch, tissue_mask=np.asarray(tissue), seed=123)
dices = []
for _ in range(8):
    L, I = sampler.sample(2, with_mri=True, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = net(standardize(I)).float().argmax(1)
    for c in (1, 2, 3):
        m = L >= 0
        inter = ((pred == c) & (L == c) & m).sum().item(); den = ((pred == c) & m).sum().item() + ((L == c) & m).sum().item()
        dices.append((c, 2 * inter / max(den, 1)))
import collections
byc = collections.defaultdict(list)
for c, d in dices: byc[c].append(d)
print("real-MRI patch Dice:", {c: round(float(np.mean(v)), 3) for c, v in byc.items()}, flush=True)
# polarity-inverted real patches (contrast-agnostic check: the target modality may have the opposite WM/GM polarity)
sampler = PatchSampler(np.asarray(labels), np.asarray(mri), patch=a.patch, tissue_mask=np.asarray(tissue), seed=123)
dinv = collections.defaultdict(list)
for _ in range(8):
    L, I = sampler.sample(2, with_mri=True, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = net(-standardize(I)).float().argmax(1)
    for c in (1, 2, 3):
        m = L >= 0
        inter = ((pred == c) & (L == c) & m).sum().item(); den = ((pred == c) & m).sum().item() + ((L == c) & m).sum().item()
        dinv[c].append(2 * inter / max(den, 1))
print("INVERTED real-MRI patch Dice:", {c: round(float(np.mean(v)), 3) for c, v in dinv.items()}, flush=True)
oct = np.load(a.work / "oct150.npy"); mask = np.load(a.work / "oct150_mask.npy")
prob = parse_volume(net, oct, patch=64, stride=32).float().numpy()
pred = prob.argmax(0)
inside = mask
print("OCT parsed class fractions inside tissue: bg %.3f WM %.3f infra %.3f supra %.3f" % tuple((pred[inside] == c).mean() for c in range(4)), flush=True)
print("OCT parsed: predicted-tissue vs mask agreement %.3f" % ((pred > 0) == inside).mean(), flush=True)
np.save(a.out / "oct150_prob.npy", prob.astype(np.float16))
print("saved", a.out / "oct150_prob.npy")

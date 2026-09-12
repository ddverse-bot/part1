#!/usr/bin/env python3
"""Run the parser over the whole-hemisphere MRI (or a crop) and cache WM / GM / tissue probability
maps as float16 (work/<parser>/mri_prob_{wm,gm,tissue}.npy) plus a Dice check against the labels (if present).

Polarity: the parser is trained on GM-bright / WM-dark ex-vivo FLASH.  For a target MRI of unknown polarity use
--auto-polarity: the parser is run on a tissue-rich sub-volume of the MRI and of its tissue-inverted copy, and the
version the parser is more confident about is used (a WM-bright MRI is inverted inside the tissue mask before parsing).
--invert forces the inversion."""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.parser import load_parser, parse_volume

ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--patch", type=int, default=128); ap.add_argument("--stride", type=int, default=96)
ap.add_argument("--auto-polarity", action="store_true", help="decide GM-bright vs WM-bright automatically (parser confidence)")
ap.add_argument("--invert", action="store_true", help="force tissue-inverted input (WM-bright MRI)")
ap.add_argument("--test-inverted-input", action="store_true", help="debug: invert the MRI first, then run the polarity logic (should re-invert)")
ap.add_argument("--out", type=Path, default=None, help="output dir (default: the parser's dir)"); ap.add_argument("--no-save", action="store_true")
a = ap.parse_args()
out = a.out or a.parser.parent; out.mkdir(parents=True, exist_ok=True)
mri = np.asarray(np.load(a.work / "mri.npy", mmap_mode="r")).astype(np.float32)
net = load_parser(a.parser)

def tissue_mask(vol):
    from skimage.filters import threshold_otsu
    thr = float(threshold_otsu(vol[::7, ::7, ::7])); return vol > thr, thr

def tissue_inverted(vol):
    m, thr = tissue_mask(vol); hi = float(np.percentile(vol[m][::13], 99.5))
    return np.where(m, np.clip(hi - vol, 0, None) + thr, 0.0).astype(np.float32)   # tissue polarity flipped, air stays 0

def confidence(vol, sub=160):
    """parser confidence on a tissue-rich sub-volume: mean max-probability inside predicted tissue, and WM fraction."""
    m, _ = tissue_mask(vol); idx = np.argwhere(m[::4, ::4, ::4]) * 4; c = idx.mean(0).astype(int)
    lo = np.clip(c - sub // 2, 0, np.array(vol.shape) - sub); hi = lo + sub
    p = parse_volume(net, vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], patch=a.patch, stride=a.stride, batch=4).float()
    tis = p[0] < 0.5; pred = p.argmax(0)
    conf = float(p.max(0).values[tis].mean()); wm = float((pred[tis] == 1).float().mean())
    # geometric polarity cue (label-free): cortex (GM) borders the background/CSF, WM does not.  With the wrong
    # polarity the parser swaps the two classes and the "WM" ends up on the outside.
    bg = (pred == 0).float()[None, None]
    near = torch.nn.functional.max_pool3d(bg, kernel_size=7, stride=1, padding=3)[0, 0] > 0.5
    gm_near = float((near & (pred >= 2)).sum() / max(int((pred >= 2).sum()), 1)); wm_near = float((near & (pred == 1)).sum() / max(int((pred == 1).sum()), 1))
    return conf, wm, gm_near - wm_near

if a.test_inverted_input:
    mri = tissue_inverted(mri); print("DEBUG: input MRI tissue-inverted", flush=True)
inverted = a.invert
if a.auto_polarity:
    c0, w0, g0 = confidence(mri); inv = tissue_inverted(mri); c1, w1, g1 = confidence(inv)
    print(f"polarity check: as-is confidence {c0:.3f} WM frac {w0:.2f} GM-outside score {g0:+.3f} | tissue-inverted confidence {c1:.3f} WM frac {w1:.2f} GM-outside score {g1:+.3f}", flush=True)
    inverted = g1 > g0                       # geometric cue decides (confidence is not discriminative)
    if inverted: mri = inv
    del inv
elif a.invert:
    mri = tissue_inverted(mri)
print("input polarity:", "tissue-inverted (WM-bright source)" if inverted else "as-is (GM-bright source)", flush=True)
t0 = time.time()
prob = parse_volume(net, mri, patch=a.patch, stride=a.stride, batch=4)   # [4,D,H,W] float16 CPU
print("parsed whole MRI in %.0fs" % (time.time() - t0), flush=True)
if not a.no_save:
    np.save(out / "mri_prob_wm.npy", prob[1].numpy())
    np.save(out / "mri_prob_gm.npy", (prob[2].float() + prob[3].float()).half().numpy())
    np.save(out / "mri_prob_tissue.npy", (1 - prob[0].float()).half().numpy())
json.dump({"inverted": bool(inverted)}, open(out / "mri_parse_polarity.json", "w"))
if (a.work / "labels4.npy").exists():
    lab = np.load(a.work / "labels4.npy", mmap_mode="r")
    pred = prob.float().argmax(0).numpy()
    res = {}
    for c, name in ((1, "WM"), (2, "infra"), (3, "supra")):
        L = np.asarray(lab) == c; P = pred == c
        res[name] = 2 * (L & P).sum() / max((L.sum() + P.sum()), 1)
    gm_L = np.isin(np.asarray(lab), (2, 3)); gm_P = np.isin(pred, (2, 3))
    res["GM"] = 2 * (gm_L & gm_P).sum() / max(gm_L.sum() + gm_P.sum(), 1)
    print("whole-hemisphere Dice vs labels (labelled region counts only where labels exist):", {k: round(float(v), 3) for k, v in res.items()})
    json.dump({k: float(v) for k, v in res.items()}, open(out / "mri_parse_dice.json", "w"), indent=1)

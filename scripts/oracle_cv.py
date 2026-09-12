#!/usr/bin/env python3
"""Cross-validated ORACLE check: is the label-free result T0 already the best affine, or is there a real residual?

The oracle affine of refine_validate.py fits the MANUAL annotations (labels + vessels) directly, so its in-sample
vessel metric is optimistic.  Here the manual MRI vessels are split into folds by connected component; the oracle
is fitted on the training folds (manual GM/WM Dice + training-fold vessels) and scored on the held-out fold
(median / fraction<150um of held-out manual MRI vessels -> nearest automatic OCT vessel).  If the held-out score
improves consistently over T0, T0 has a genuine residual of that size; if not, T0 is at the annotation noise floor.
Also fits rigid-only and labels-only oracles to attribute the gain (translation/rotation vs scale; labels vs vessels).
Evaluation only: nothing here is part of the registration method.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel
from octreg.refine import Refiner, compose, params_from_matrix
from octreg.features import build_features
from octreg.evaluate import transform_diff

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T0", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--folds", type=int, default=2); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--iters", type=int, default=300); ap.add_argument("--cap-mm", type=float, default=0.6)
ap.add_argument("--extra-T", type=str, default="", help="comma-separated extra transforms to score on the same folds (name=path)")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
def say(*x): print(*x, flush=True)

A_mri = np.load(a.work / "mri_affine.npy"); mri_shape = np.load(a.work / "mri.npy", mmap_mode="r").shape
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); ba = np.load(a.work / "ba_labels.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
T0 = np.load(a.T0)
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; c_o_t = to_t(c_o)
centre = (T0 @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri_shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]; A_REG = to_t(A_reg)
lab_reg = np.asarray(labels4[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); ba_reg = np.asarray(ba[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
lab_use = np.where(ba_reg > 0, ba_reg, lab_reg)
LAB_WM = to_t((lab_use == 1).astype(np.float32))[None]; LAB_GM = to_t(np.isin(lab_use, (2, 3)).astype(np.float32))[None]

# OCT side: Otsu tissue classes (polarity from parser) as in the method; anchor gives the sample points/weights
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
_pred = oct_prob.argmax(0); wm_bright = bool(oct150[(_pred == 1) & oct_mask].mean() > oct150[(_pred >= 2) & oct_mask].mean())
FO = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=wm_bright)
anchor = Refiner(FM, A_reg, FO, A_oct, to_t(oct_mask)[None].float())

# manual vessels (region), OCT automatic vessels (12 um Frangi, EDT at 48 um)
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); man = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
comp, n_comp = ndimage.label(man); say(f"manual vessel voxels in region {int(man.sum())}, components {n_comp}")
d0um = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
EDT_O = to_t(d0um / 1000.0)[None]; A_O48 = to_t(A0); oct_shape48 = to_t(np.array(d0um.shape) - 1)
vox = np.argwhere(d0um <= 0.0); rng = np.random.default_rng(0)
vox = vox[rng.choice(len(vox), size=min(60000, len(vox)), replace=False)]
oct_ves_pts = to_t((A0 @ np.c_[vox, np.ones(len(vox))].T).T[:, :3])
man_ijk = np.argwhere(man); man_comp = comp[man]                                # component id per manual voxel
man_w_all = (A_reg @ np.c_[man_ijk, np.ones(len(man_ijk))].T).T[:, :3]

def cap(d): return torch.clamp(d, max=a.cap_mm)

def make_oracle(train_mask, use_vessels=True, use_labels=True):
    """oracle objective restricted to training-fold manual vessels."""
    m = np.zeros_like(man); m[tuple(man_ijk[train_mask].T)] = True
    edt = ndimage.distance_transform_edt(~m, sampling=(0.15,) * 3).astype(np.float32); EDT_TR = to_t(edt)[None]
    tr_w = to_t(man_w_all[train_mask])
    def loss(T):
        L = 0.0
        if use_labels:
            pts = anchor.pts_o; w = anchor.w; pm = apply_affine(T, pts)
            wm_m = sample_at_world(LAB_WM, A_REG, pm)[0]; gm_m = sample_at_world(LAB_GM, A_REG, pm)[0]
            dice = 0.0
            for po, pm_ in ((anchor.fo[0], wm_m), (anchor.fo[1], gm_m)):
                dice = dice + 2 * (w * po * pm_).sum() / ((w * po).sum() + (w * pm_).sum() + 1e-6)
            L = L + 1 - dice / 2
        if use_vessels:
            p_mv = apply_affine(T, oct_ves_pts); l_ov = cap(sample_at_world(EDT_TR, A_REG, p_mv, padding=a.cap_mm)[0]).mean()
            p_o = apply_affine(torch.linalg.inv(T), tr_w); v = apply_affine(torch.linalg.inv(A_O48), p_o)
            inside = ((v >= 0) & (v <= oct_shape48)).all(1)
            l_mo = (cap(sample_at_world(EDT_O, A_O48, p_o)[0]) * inside).sum() / inside.float().sum().clamp(min=1)
            L = L + l_ov + l_mo
        return L
    return loss

def score(T, test_mask):
    """held-out manual MRI vessels -> nearest OCT vessel (um): median, frac<150, frac<300, n inside block."""
    p_o = apply_affine(torch.linalg.inv(to_t(T)), to_t(man_w_all[test_mask])); v = apply_affine(torch.linalg.inv(A_O48), p_o)
    inside = ((v >= 0) & (v <= oct_shape48)).all(1)
    d = sample_at_world(EDT_O, A_O48, p_o)[0][inside] * 1000
    return {"median_um": float(d.median()), "f150": float((d <= 150).float().mean()), "f300": float((d <= 300).float().mean()), "n": int(inside.sum())}

def optimize(T_init, loss_fn, dof="affine", iters=300, lr_rot=0.005, lr_t=0.1, lr_ls=0.005, lr_sh=0.005, ls_clamp=0.25, sh_clamp=0.25, reg=1.0):
    r0, t0, ls0, sh0, mirror = params_from_matrix(T_init, c_o)
    r = to_t(r0).requires_grad_(True); t = to_t(t0).requires_grad_(True); ls = to_t(ls0); sh = to_t(sh0)
    groups = [{"params": [r], "lr": lr_rot}, {"params": [t], "lr": lr_t}]
    if dof == "affine":
        ls.requires_grad_(True); sh.requires_grad_(True); groups += [{"params": [ls], "lr": lr_ls}, {"params": [sh], "lr": lr_sh}]
    opt = torch.optim.Adam(groups); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 0.0); best = (1e9, None)
    for it in range(iters):
        opt.zero_grad(); T = compose(r, t, ls, sh, c_o_t, mirror); L = loss_fn(T)
        tot = L + (reg * ((ls ** 2).sum() + (sh ** 2).sum()) if dof == "affine" else 0.0)
        tot.backward(); opt.step(); sched.step()
        if dof == "affine":
            with torch.no_grad(): ls.clamp_(-ls_clamp, ls_clamp); sh.clamp_(-sh_clamp, sh_clamp)
        if L.item() < best[0]: best = (L.item(), T.detach().cpu().numpy().copy())
    return best[1]

variants = [("oracle affine (labels+vessels)", dict(dof="affine", use_vessels=True, use_labels=True)),
            ("oracle rigid  (labels+vessels)", dict(dof="rigid", use_vessels=True, use_labels=True)),
            ("oracle affine (vessels only)", dict(dof="affine", use_vessels=True, use_labels=False)),
            ("oracle affine (labels only)", dict(dof="affine", use_vessels=False, use_labels=True))]
extra = {}
for kv in [s for s in a.extra_T.split(",") if s]:
    k, v = kv.split("="); extra[k] = np.load(v)
results = {"T0": [], **{n: [] for n, _ in variants}, **{k: [] for k in extra}}
moves = {n: [] for n, _ in variants}
t_start = time.time()
for seed in range(a.seeds):
    fold_of_comp = np.random.default_rng(seed).integers(0, a.folds, size=n_comp + 1)
    for f in range(a.folds):
        test = fold_of_comp[man_comp] == f; train = ~test
        results["T0"].append(score(T0, test))
        for k, Tk in extra.items(): results[k].append(score(Tk, test))
        for name, cfg in variants:
            loss = make_oracle(train, use_vessels=cfg["use_vessels"], use_labels=cfg["use_labels"])
            T = optimize(T0, loss, dof=cfg["dof"], iters=a.iters)
            results[name].append(score(T, test)); d = transform_diff(T, T0, c_o); moves[name].append(d)
        s0 = results["T0"][-1]; s1 = results[variants[0][0]][-1]
        say(f"seed {seed} fold {f}: held-out n {s0['n']} | T0 med {s0['median_um']:.0f} f150 {s0['f150']:.2f} | oracle affine med {s1['median_um']:.0f} f150 {s1['f150']:.2f} | "
            + " | ".join(f"{n.split('(')[0].strip()}({n.split('(')[1][:-1]}) med {results[n][-1]['median_um']:.0f} f150 {results[n][-1]['f150']:.2f}" for n, _ in variants[1:])
            + f" | {time.time()-t_start:.0f}s")

summary = {}
for k, rows in results.items():
    med = np.array([r["median_um"] for r in rows]); f150 = np.array([r["f150"] for r in rows]); f300 = np.array([r["f300"] for r in rows])
    summary[k] = {"median_um_mean": float(med.mean()), "median_um_sd": float(med.std()), "f150_mean": float(f150.mean()), "f150_sd": float(f150.std()), "f300_mean": float(f300.mean())}
    if k in moves and moves[k]:
        summary[k]["vs_T0_centre_mm"] = float(np.mean([d["centre_mm"] for d in moves[k]])); summary[k]["vs_T0_corner_mm"] = float(np.mean([d["corner_mean_mm"] for d in moves[k]]))
say("\n== held-out manual-vessel score, mean over folds x seeds (lower median / higher f150 = better)")
for k, s in summary.items():
    say(f"{k:34s} median {s['median_um_mean']:.0f} +- {s['median_um_sd']:.0f} um   f150 {s['f150_mean']:.3f} +- {s['f150_sd']:.3f}   f300 {s['f300_mean']:.3f}" + (f"   moved vs T0: centre {s['vs_T0_centre_mm']:.2f} mm corners {s['vs_T0_corner_mm']:.2f} mm" if "vs_T0_centre_mm" in s else ""))
json.dump({"T0": str(a.T0), "folds": a.folds, "seeds": a.seeds, "per_fold": results, "summary": summary}, open(a.out / "oracle_cv.json", "w"), indent=1, default=float)
say("saved", a.out / "oracle_cv.json")

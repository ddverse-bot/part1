#!/usr/bin/env python3
"""Probe the anisotropic-scale question: along which OCT axis does the oracle stretch, does the label-free objective
(mixed features NCC) see it, does the manual-label Dice see it, and do the manual vessels see it?

    profiles of  NCC(mixed) / Dice(manual labels) / vessel score(manual MRI vessels -> OCT vessels)
    as a function of a scale factor s applied along ONE OCT array axis about the block centre, on top of T0.
Then: label-free affine refinement from T0 with NO scale regularisation / wide clamp -> where does it go?
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel
from octreg.refine import Refiner, params_from_matrix
from octreg.features import build_features
from octreg.evaluate import gm_wm_overlap, transform_diff

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T0", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--Tref", type=Path, default=None, help="oracle transform for comparison")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
def say(*x): print(*x, flush=True)

A_mri = np.load(a.work / "mri_affine.npy"); mri_shape = np.load(a.work / "mri.npy", mmap_mode="r").shape
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); ba = np.load(a.work / "ba_labels.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32); T0 = np.load(a.T0)
sh = np.array(oct150.shape); c_o = (A_oct @ np.r_[(sh - 1) / 2.0, 1.0])[:3]
centre = (T0 @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri_shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]; A_REG = to_t(A_reg)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
_pred = oct_prob.argmax(0); wm_bright = bool(oct150[(_pred == 1) & oct_mask].mean() > oct150[(_pred >= 2) & oct_mask].mean())
FO = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=wm_bright)
anchor = Refiner(FM, A_reg, FO, A_oct, to_t(oct_mask)[None].float())
oct_class = np.where(oct_mask, np.where(FO[0].cpu().numpy() > 0.5, 1, 2), 0)
lab_reg = np.asarray(labels4[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); ba_reg = np.asarray(ba[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
lab_use = np.where(ba_reg > 0, ba_reg, lab_reg)
LAB_WM = to_t((lab_use == 1).astype(np.float32))[None]; LAB_GM = to_t(np.isin(lab_use, (2, 3)).astype(np.float32))[None]
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); man = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
man_w = to_t((A_reg @ np.c_[np.argwhere(man), np.ones(int(man.sum()))].T).T[:, :3])
d0um = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
EDT_O = to_t(d0um / 1000.0)[None]; A_O48 = to_t(A0); oct_shape48 = to_t(np.array(d0um.shape) - 1)

def ncc_mixed(T):
    with torch.no_grad(): return float(1 - anchor.evaluate(T))
def dice_manual(T):
    with torch.no_grad():
        pm = apply_affine(to_t(T), anchor.pts_o); w = anchor.w
        wm_m = sample_at_world(LAB_WM, A_REG, pm)[0]; gm_m = sample_at_world(LAB_GM, A_REG, pm)[0]
        out = []
        for po, pm_ in ((anchor.fo[0], wm_m), (anchor.fo[1], gm_m)):
            out.append(float(2 * (w * po * pm_).sum() / ((w * po).sum() + (w * pm_).sum() + 1e-6)))
        return out
def vessel_score(T):
    with torch.no_grad():
        p_o = apply_affine(torch.linalg.inv(to_t(T)), man_w); v = apply_affine(torch.linalg.inv(A_O48), p_o)
        inside = ((v >= 0) & (v <= oct_shape48)).all(1); d = sample_at_world(EDT_O, A_O48, p_o)[0][inside] * 1000
        return float(d.median()), float((d <= 150).float().mean()), int(inside.sum())

axis_names = ["i (627-axis, depth/stack)", "j (1271-axis)", "k (1230-axis)"]
if a.Tref is not None:
    Tref = np.load(a.Tref)
    # residual at block corners in OCT array frame
    corners = np.array([[i, j, k] for i in (0, sh[0] - 1) for j in (0, sh[1] - 1) for k in (0, sh[2] - 1)], float)
    pw = (A_oct @ np.c_[corners, np.ones(8)].T).T
    d = (np.linalg.inv(T0) @ Tref @ pw.T).T[:, :3] - pw[:, :3]
    R = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
    say("T0 -> Tref residual at the 8 block corners, components along OCT array axes i,j,k (mm):")
    for c, r in zip(corners, d @ R): say("  corner", c.astype(int).tolist(), np.round(r, 2).tolist())
    M = np.linalg.inv(T0[:3, :3]) @ Tref[:3, :3]
    # relative stretch along each array axis: |M R_col| / |R_col|
    say("relative stretch of Tref vs T0 along OCT array axes i,j,k:", [round(float(np.linalg.norm(M @ R[:, q])), 3) for q in range(3)])
    say("T0 scales along i,j,k:", [round(float(np.linalg.norm(T0[:3, :3] @ R[:, q])), 3) for q in range(3)], " Tref:", [round(float(np.linalg.norm(Tref[:3, :3] @ R[:, q])), 3) for q in range(3)])

def scaled(T, axis, s):
    """T composed with an axis-aligned OCT-array-frame scale s about the block centre (applied in OCT world before T)."""
    R = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
    S = np.eye(3) + (s - 1) * np.outer(R[:, axis], R[:, axis])
    Sw = np.eye(4); Sw[:3, :3] = S; Sw[:3, 3] = c_o - S @ c_o
    return T @ Sw

prof = {}
for axis in range(3):
    rows = []
    for s in np.round(np.arange(0.85, 1.301, 0.025), 3):
        T = scaled(T0, axis, s); n = ncc_mixed(T); dw, dg = dice_manual(T); vm, vf, vn = vessel_score(T)
        rows.append([float(s), n, dw, dg, vm, vf]);
    prof[axis_names[axis]] = rows
    say(f"\n== scale along {axis_names[axis]}  (s, NCC-mixed, Dice WM, Dice GM, vessel median um, f150)")
    for r in rows: say("  " + "  ".join(f"{x:.3f}" if i < 4 else f"{x:.2f}" for i, x in enumerate(r)))
    b = max(rows, key=lambda r: r[1]); bd = max(rows, key=lambda r: r[2] + r[3]); bv = max(rows, key=lambda r: r[5])
    say(f"  argmax: NCC-mixed s={b[0]:.3f} | manual Dice s={bd[0]:.3f} | vessels f150 s={bv[0]:.3f}")

# label-free affine refinement from T0 without the scale prior
say("\n== label-free affine refinement from T0 (mixed features), no log-scale/shear regularisation, clamp 0.35")
Tf, l = anchor.refine(T0, dof="affine", iters=300, ls_clamp=0.35, sh_clamp=0.35, reg=0.0)
R = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
say("  scales along i,j,k:", [round(float(np.linalg.norm(Tf[:3, :3] @ R[:, q])), 3) for q in range(3)], " ncc", round(1 - l, 4), " vs T0", transform_diff(Tf, T0, c_o), " vessels", vessel_score(Tf), " Dice manual", dice_manual(Tf))
np.save(a.out / "T_free_affine.npy", Tf)
json.dump(prof, open(a.out / "scale_profiles.json", "w"), indent=1)

#!/usr/bin/env python3
"""Label-free non-rigid stage on top of the vascular affine, and its validation against the manual annotations.

Variants: control grid / smoothness / with-without the vessel channel; plus an ORACLE non-rigid (manual labels + manual
vessels, evaluation only) as the ceiling.  Scores: held-out manual MRI vessels -> OCT vessels (through the inverse of
the non-rigid map), Dice of OCT tissue classes vs manual labels, displacement statistics."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel
from octreg.features import build_features
from octreg.evaluate import transform_diff
from octreg.vascular import mri_dark_channel, oct_vessel_mask, density_on_grid
from octreg.nonrigid import GridDisp, nonrigid_refine, inverse_points, loss_with_disp
from octreg.refine import Refiner

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T", type=Path, required=True, help="vascular affine (start)"); ap.add_argument("--T-struct", type=Path, default=None)
ap.add_argument("--T-oracle", type=Path, default=None); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--iters", type=int, default=300)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
def say(*x): print(*x, flush=True)

A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); ba = np.load(a.work / "ba_labels.npy", mmap_mode="r"); tissue = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32); T = np.load(a.T)
sh = np.array(oct150.shape); c_o = (A_oct @ np.r_[(sh - 1) / 2.0, 1.0])[:3]; centre = (T @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]; A_REG = to_t(A_reg)
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); tis_reg = np.asarray(tissue[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
FM2 = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
_pred = oct_prob.argmax(0); wm_bright = bool(oct150[(_pred == 1) & oct_mask].mean() > oct150[(_pred >= 2) & oct_mask].mean())
FO2 = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=wm_bright); MO = to_t(oct_mask)[None].float()
mri_v = mri_dark_channel(mri_reg, tis_reg)
oct24 = np.load(a.work / "oct24.npy").astype(np.float32); m24 = np.load(a.work / "oct24_mask.npy"); A24 = np.load(a.work / "oct24_affine.npy")
oct_v = density_on_grid(oct_vessel_mask(oct24, m24, q=99.0, sigmas=(1.0, 1.5, 2.2, 3.0, 4.4)), A24, A_oct, oct150.shape, oct_mask, pool=6); del oct24
FM3 = torch.cat([FM2, mri_v[None]], 0); FO3 = torch.cat([FO2, oct_v[None]], 0)

# manual annotations (evaluation / oracle only)
lab_reg = np.asarray(labels4[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); ba_reg = np.asarray(ba[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
lab_use = np.where(ba_reg > 0, ba_reg, lab_reg)
LAB_WM = to_t((lab_use == 1).astype(np.float32))[None]; LAB_GM = to_t(np.isin(lab_use, (2, 3)).astype(np.float32))[None]
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); man = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
man_w = to_t((A_reg @ np.c_[np.argwhere(man), np.ones(int(man.sum()))].T).T[:, :3])
edt_man = ndimage.distance_transform_edt(~man, sampling=(0.15,) * 3).astype(np.float32); EDT_MAN = to_t(edt_man)[None]
d0um = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
EDT_O = to_t(d0um / 1000.0)[None]; A_O48 = to_t(A0); oct_shape48 = to_t(np.array(d0um.shape) - 1)
vox = np.argwhere(d0um <= 0.0); rng = np.random.default_rng(0); vox = vox[rng.choice(len(vox), size=min(60000, len(vox)), replace=False)]
oct_ves_pts = to_t((A0 @ np.c_[vox, np.ones(len(vox))].T).T[:, :3])
anchor = Refiner(FM2, A_reg, FO2, A_oct, MO)      # sample points / weights / OCT classes for the Dice evaluation
wm_o = (anchor.fo[0] > 0.5).float(); gm_o = (anchor.fo[1] > 0.5).float(); w = anchor.w

@torch.no_grad()
def evaluate(T_, disp=None, name=""):
    Tt = to_t(T_)
    # manual MRI vessels -> OCT (inverse map) -> nearest OCT vessel
    p_o = inverse_points(T_, disp, man_w) if disp is not None else apply_affine(torch.linalg.inv(Tt), man_w)
    v = apply_affine(torch.linalg.inv(A_O48), p_o); inside = ((v >= 0) & (v <= oct_shape48)).all(1)
    d = sample_at_world(EDT_O, A_O48, p_o)[0][inside] * 1000
    vm, vf = float(d.median()), float((d <= 150).float().mean())
    # Dice OCT classes vs manual labels (forward map of OCT tissue points)
    pm = apply_affine(Tt, anchor.pts_o) + (disp(anchor.pts_o) if disp is not None else 0)
    wm_m = (sample_at_world(LAB_WM, A_REG, pm)[0] > 0.5).float(); gm_m = (sample_at_world(LAB_GM, A_REG, pm)[0] > 0.5).float()
    dwm = float(2 * (w * wm_o * wm_m).sum() / ((w * wm_o).sum() + (w * wm_m).sum() + 1e-6)); dgm = float(2 * (w * gm_o * gm_m).sum() / ((w * gm_o).sum() + (w * gm_m).sum() + 1e-6))
    # OCT vessels -> manual MRI vessel EDT (forward)
    pv = apply_affine(Tt, oct_ves_pts) + (disp(oct_ves_pts) if disp is not None else 0)
    dov = sample_at_world(EDT_MAN, A_REG, pv, padding=5.0)[0] * 1000
    extra = ""
    if disp is not None:
        mag = disp(anchor.pts_o).norm(dim=1); extra = f" | disp mean {mag.mean():.3f} p95 {torch.quantile(mag,0.95):.3f} max {mag.max():.3f} mm"
    say(f"{name:48s} vessels med {vm:4.0f} f150 {vf:.2f} | Dice WM {dwm:.3f} GM {dgm:.3f} | OCTves->manual med {dov.median():.0f}{extra}")
    return {"name": name, "vessel_median_um": vm, "vessel_f150": vf, "dice_WM": dwm, "dice_GM": dgm, "octves2manual_median_um": float(dov.median()),
            **({"disp_mean_mm": float(mag.mean()), "disp_p95_mm": float(torch.quantile(mag, 0.95)), "disp_max_mm": float(mag.max())} if disp is not None else {})}

rows = []
if a.T_struct: rows.append(evaluate(np.load(a.T_struct), None, "structural affine"))
rows.append(evaluate(T, None, "vascular affine (start)"))
if a.T_oracle: rows.append(evaluate(np.load(a.T_oracle), None, "oracle affine (manual)"))

t0 = time.time()
for grid, lam_s, lam_m, cw, tag in [((5, 8, 8), 5.0, 0.5, (1, 1, 1), "grid 5x8x8 smooth 5"),
                                    ((5, 8, 8), 20.0, 0.5, (1, 1, 1), "grid 5x8x8 smooth 20"),
                                    ((5, 8, 8), 1.0, 0.5, (1, 1, 1), "grid 5x8x8 smooth 1"),
                                    ((4, 6, 6), 5.0, 0.5, (1, 1, 1), "grid 4x6x6 smooth 5"),
                                    ((7, 12, 12), 5.0, 0.5, (1, 1, 1), "grid 7x12x12 smooth 5"),
                                    ((5, 8, 8), 5.0, 0.5, (1, 1, 0), "grid 5x8x8 smooth 5, NO vessel channel")]:
    disp, info = nonrigid_refine(T, FM3, A_reg, FO3, A_oct, MO, grid=grid, iters=a.iters, lam_smooth=lam_s, lam_mag=lam_m, chan_w=cw)
    r = evaluate(T, disp, f"label-free nonrigid: {tag}"); r["ncc_channels"] = info["ncc_channels"]; rows.append(r)
    np.save(a.out / f"disp_{tag.replace(' ', '_').replace(',', '')}.npy", disp.state())
say(f"[label-free variants {time.time()-t0:.0f}s]")

# ORACLE non-rigid (manual labels Dice + manual vessels both directions), from the vascular affine — the ceiling
def oracle_loss(disp):
    pm = apply_affine(to_t(T), anchor.pts_o) + disp(anchor.pts_o)
    wm_m = sample_at_world(LAB_WM, A_REG, pm)[0]; gm_m = sample_at_world(LAB_GM, A_REG, pm)[0]
    dice = 0.0
    for po, pm_ in ((anchor.fo[0], wm_m), (anchor.fo[1], gm_m)):
        dice = dice + 2 * (w * po * pm_).sum() / ((w * po).sum() + (w * pm_).sum() + 1e-6)
    p_mv = apply_affine(to_t(T), oct_ves_pts) + disp(oct_ves_pts)
    l_ov = torch.clamp(sample_at_world(EDT_MAN, A_REG, p_mv, padding=0.6)[0], max=0.6).mean()
    p_o = inverse_points(T, disp, man_w, n_iter=3); v = apply_affine(torch.linalg.inv(A_O48), p_o); inside = ((v >= 0) & (v <= oct_shape48)).all(1)
    l_mo = (torch.clamp(sample_at_world(EDT_O, A_O48, p_o)[0], max=0.6) * inside).sum() / inside.float().sum().clamp(min=1)
    return (1 - dice / 2) + l_ov + l_mo
for lam_s, tag in [(5.0, "smooth 5"), (1.0, "smooth 1")]:
    disp = GridDisp(oct150.shape, A_oct, (5, 8, 8)); opt = torch.optim.Adam([disp.g], lr=0.05)
    for it in range(a.iters):
        opt.zero_grad(); L = oracle_loss(disp) + lam_s * disp.smooth() + 0.5 * disp.magnitude(); L.backward(); opt.step()
    rows.append(evaluate(T, disp, f"ORACLE nonrigid (manual): grid 5x8x8 {tag}"))
json.dump({"T": str(a.T), "rows": rows}, open(a.out / "nonrigid_stage.json", "w"), indent=1, default=float)
say("saved", a.out / "nonrigid_stage.json")

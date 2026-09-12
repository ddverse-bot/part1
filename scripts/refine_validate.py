#!/usr/bin/env python3
"""Second-stage refinement + "is this the best?" validation, starting from a registration result T0.

  A. Automatic (annotation-free) vascular refinement:
        OCT vessels  = automatic Frangi segmentation of the 12 um OCT (the DANDI ves_seg is that)
        MRI vessels  = automatic dark-tube vesselness (Frangi) on the 0.15 mm MRI, top-q inside brain tissue
        objective    = robust symmetric point-to-vessel distance (both directions, capped) + tissue-class NCC anchor
        transform    = affine (constrained), Adam
  B. Oracle references (use the MANUAL annotations; evaluation only, never part of the method):
        B1 affine    : Costantini-style objective on manual labels (soft Dice of OCT tissue classes vs manual GM/WM
                       labels + manual-vessel MED both directions), from T0
        B2 nonrigid  : same objective + a smooth displacement grid over the block (free-form), from B1
  All variants are scored with the held-out manual metrics (manual MRI vessels -> OCT vessels, Dice vs manual labels)
  and compared to T0.  Writes result JSON + transforms.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, avg_pool_iso, save_nifti
from octreg.refine import Refiner, compose, params_from_matrix
from octreg.features import build_features, otsu_two_class, frangi_dark_vesselness
from octreg.evaluate import vessel_distance_stats, gm_wm_overlap, transform_diff

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T0", type=Path, required=True, help="starting transform .npy (OCT world -> MRI world)")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--vessel-q", type=float, default=99.7, help="MRI vesselness percentile (inside tissue) for automatic vessel candidates")
ap.add_argument("--cap-mm", type=float, default=0.6, help="robust cap on point-to-vessel distances")
ap.add_argument("--w-ncc", type=float, default=1.0); ap.add_argument("--w-ves", type=float, default=1.0)
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--grid", type=str, default="5,8,8", help="nonrigid oracle control grid (z,y,x)")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"

def say(*x): print(*x, flush=True)

# ---------------------------------------------------------------- data
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); ba = np.load(a.work / "ba_labels.npy", mmap_mode="r")
tissue = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32)
T0 = np.load(a.T0)
c_o = (A_oct @ np.r_[(np.array(oct150.shape) - 1) / 2.0, 1.0])[:3]; c_o_t = to_t(c_o)
centre = (T0 @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
tis_reg = np.asarray(tissue[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
lab_reg = np.asarray(labels4[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); ba_reg = np.asarray(ba[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
say(f"region {mri_reg.shape} around {np.round(centre,1).tolist()}")

# tissue-class features (anchor): MRI parser probs, OCT Otsu classes (polarity from parser)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
FM = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
_pred = oct_prob.argmax(0); wm_bright = bool(oct150[(_pred == 1) & oct_mask].mean() > oct150[(_pred >= 2) & oct_mask].mean())
FO = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=wm_bright)
MO = to_t(oct_mask)[None].float()
anchor = Refiner(FM, A_reg, FO, A_oct, MO)
oct_class_otsu = np.where(oct_mask, np.where(FO[0].cpu().numpy() > 0.5, 1, 2), 0)

# manual structures (evaluation / oracle)
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); ves_ijk = np.argwhere(np.asarray(mri_ves))
ves_w = (A_mri @ np.c_[ves_ijk, np.ones(len(ves_ijk))].T).T[:, :3]
d0um = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
EDT_O = to_t(d0um / 1000.0)[None]; A_O48 = to_t(A0); oct_shape48 = to_t(np.array(d0um.shape) - 1)
vox = np.argwhere(d0um <= 0.0); rng = np.random.default_rng(0)
vox = vox[rng.choice(len(vox), size=min(60000, len(vox)), replace=False)]
oct_ves_pts = to_t((A0 @ np.c_[vox, np.ones(len(vox))].T).T[:, :3])         # OCT world (automatic Frangi @12um)

# ---------------------------------------------------------------- automatic MRI vessel candidates (annotation-free)
vess = frangi_dark_vesselness(to_t(mri_reg)[None])[0] * to_t(tis_reg.astype(np.float32))
thr = float(np.percentile(vess[to_t(tis_reg.astype(bool), dtype=torch.bool)].cpu().numpy(), a.vessel_q))
cand = (vess > thr).cpu().numpy()
lab_c, n_c = ndimage.label(cand)
sizes = ndimage.sum(cand, lab_c, index=np.arange(1, n_c + 1))
keep = np.isin(lab_c, np.arange(1, n_c + 1)[sizes >= 4])           # drop tiny specks
cand = keep
say(f"automatic MRI vessel candidates: {int(cand.sum())} voxels ({n_c} components, kept size>=4)")
edt_auto = ndimage.distance_transform_edt(~cand, sampling=(0.15,) * 3).astype(np.float32)
EDT_AUTO = to_t(edt_auto)[None]; A_REG = to_t(A_reg)
cand_ijk = np.argwhere(cand); cand_w = to_t((A_reg @ np.c_[cand_ijk, np.ones(len(cand_ijk))].T).T[:, :3])
# manual MRI vessel EDT (oracle)
man = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
edt_man = ndimage.distance_transform_edt(~man, sampling=(0.15,) * 3).astype(np.float32); EDT_MAN = to_t(edt_man)[None]
man_w = to_t((A_reg @ np.c_[np.argwhere(man), np.ones(int(man.sum()))].T).T[:, :3])
# manual GM/WM label maps as soft targets for the oracle (BA labels where present, else whole-hemi labels)
lab_use = np.where(ba_reg > 0, ba_reg, lab_reg)
LAB_WM = to_t((lab_use == 1).astype(np.float32))[None]; LAB_GM = to_t(np.isin(lab_use, (2, 3)).astype(np.float32))[None]
# precision check of the automatic candidates vs manual labels (how many candidates lie within 0.3 mm of a manual vessel)
d_c = sample_at_world(EDT_MAN, A_REG, cand_w)[0]
say(f"automatic MRI vessel candidates within 0.3 mm of a manual vessel: {(d_c <= 0.3).float().mean().item():.2f}; manual vessels within 0.3 mm of a candidate: {(sample_at_world(EDT_AUTO, A_REG, man_w)[0] <= 0.3).float().mean().item():.2f}")

# ---------------------------------------------------------------- objectives
def cap(d): return torch.clamp(d, max=a.cap_mm)

def vessel_auto_loss(T):
    """symmetric robust distance: OCT auto vessels -> MRI auto vessel EDT ; MRI auto candidates (inside block) -> OCT vessel EDT."""
    p_m = apply_affine(T, oct_ves_pts); l1 = cap(sample_at_world(EDT_AUTO, A_REG, p_m, padding=a.cap_mm)[0]).mean()
    Tinv = torch.linalg.inv(T); p_o = apply_affine(Tinv, cand_w); v = apply_affine(torch.linalg.inv(A_O48), p_o)
    inside = ((v >= 0) & (v <= oct_shape48)).all(1)
    d = cap(sample_at_world(EDT_O, A_O48, p_o)[0]); l2 = (d * inside).sum() / inside.float().sum().clamp(min=1)
    return l1 + l2

def oracle_loss(T, disp=None):
    """manual labels: soft Dice (OCT Otsu classes vs manual WM/GM) + manual vessel MED both directions."""
    pts = anchor.pts_o; w = anchor.w
    pm = apply_affine(T, pts)
    if disp is not None: pm = pm + disp(pts)
    wm_m = sample_at_world(LAB_WM, A_REG, pm)[0]; gm_m = sample_at_world(LAB_GM, A_REG, pm)[0]
    wm_o = anchor.fo[0]; gm_o = anchor.fo[1]
    dice = 0.0
    for po, pm_ in ((wm_o, wm_m), (gm_o, gm_m)):
        inter = (w * po * pm_).sum(); den = (w * po).sum() + (w * pm_).sum()
        dice = dice + 2 * inter / (den + 1e-6)
    l_dice = 1 - dice / 2
    p_mv = apply_affine(T, oct_ves_pts)
    if disp is not None: p_mv = p_mv + disp(oct_ves_pts)
    l_ov = cap(sample_at_world(EDT_MAN, A_REG, p_mv, padding=a.cap_mm)[0]).mean()
    Tinv = torch.linalg.inv(T); p_o = apply_affine(Tinv, man_w)         # (no inverse for disp; affine only for this term)
    v = apply_affine(torch.linalg.inv(A_O48), p_o); inside = ((v >= 0) & (v <= oct_shape48)).all(1)
    l_mo = (cap(sample_at_world(EDT_O, A_O48, p_o)[0]) * inside).sum() / inside.float().sum().clamp(min=1)
    return l_dice + l_ov + l_mo

def optimize_affine(T_init, loss_fn, iters, lr_rot=0.005, lr_t=0.1, lr_ls=0.005, lr_sh=0.005, ls_clamp=0.2, sh_clamp=0.2, reg=1.0):
    r0, t0, ls0, sh0, mirror = params_from_matrix(T_init, c_o)
    r = to_t(r0).requires_grad_(True); t = to_t(t0).requires_grad_(True); ls = to_t(ls0).requires_grad_(True); sh = to_t(sh0).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [r], "lr": lr_rot}, {"params": [t], "lr": lr_t}, {"params": [ls], "lr": lr_ls}, {"params": [sh], "lr": lr_sh}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 0.0); best = (1e9, None)
    for it in range(iters):
        opt.zero_grad(); T = compose(r, t, ls, sh, c_o_t, mirror); L = loss_fn(T)
        (L + reg * ((ls ** 2).sum() + (sh ** 2).sum())).backward(); opt.step(); sched.step()
        with torch.no_grad(): ls.clamp_(-ls_clamp, ls_clamp); sh.clamp_(-sh_clamp, sh_clamp)
        if L.item() < best[0]: best = (L.item(), T.detach().cpu().numpy().copy())
    return best[1], best[0]

class GridDisp:
    """Smooth free-form displacement (mm) over the OCT block: control grid (gz,gy,gx), trilinear interpolation."""
    def __init__(self, shape_oct, A_oct_, grid_shape):
        self.g = torch.zeros((1, 3, *grid_shape), device=dev, requires_grad=True)
        self.A_inv = torch.linalg.inv(to_t(A_oct_)); self.size = to_t(np.array(shape_oct) - 1)
    def __call__(self, pts_world):
        v = apply_affine(self.A_inv, pts_world) / self.size * 2 - 1                # OCT voxel coords normalized
        grid = v[..., [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        return F.grid_sample(self.g, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(3, -1).T
    def smooth(self):
        g = self.g
        return ((g[..., 1:, :, :] - g[..., :-1, :, :]) ** 2).mean() + ((g[..., :, 1:, :] - g[..., :, :-1, :]) ** 2).mean() + ((g[..., :, :, 1:] - g[..., :, :, :-1]) ** 2).mean()

def evaluate(T, name, disp=None):
    ev = vessel_distance_stats(T, ves_ijk, A_mri, d0um, A0, n_ctrl=0)["registered"]
    gw = gm_wm_overlap(T, oct_class_otsu, oct_mask, A_oct, np.asarray(labels4), A_mri)
    d = transform_diff(T, T0, c_o)
    with torch.no_grad():
        ncc = 1 - anchor.evaluate(T); vl = float(vessel_auto_loss(to_t(T))); ol = float(oracle_loss(to_t(T)))
    row = {"name": name, "vessel_median_um": ev.get("median_um"), "vessel_p75_um": ev.get("p75_um"), "vessel_f150": ev.get("frac_within_150um"), "vessel_f300": ev.get("frac_within_300um"),
           "n_inside": ev.get("n_inside"), "dice_WM": gw["dice_WM"], "dice_GM": gw["dice_GM"], "ncc_anchor": ncc, "auto_vessel_loss": vl, "oracle_loss": ol,
           "vs_T0_centre_mm": d["centre_mm"], "vs_T0_rot_deg": d["rotation_deg"], "vs_T0_corner_mm": d["corner_mean_mm"], "scales": np.linalg.norm(T[:3, :3], axis=0).tolist()}
    say(f"{name:26s} ves med {row['vessel_median_um']:.0f} p75 {row['vessel_p75_um']:.0f} f150 {row['vessel_f150']:.2f} | Dice WM {row['dice_WM']:.3f} GM {row['dice_GM']:.3f} | ncc {ncc:.3f} autoV {vl:.3f} oracle {ol:.3f} | vs T0: {d['centre_mm']:.2f} mm {d['rotation_deg']:.2f} deg corners {d['corner_mean_mm']:.2f} mm")
    return row

rows = []
rows.append(evaluate(T0, "T0 (label-free result)"))
# A. automatic vascular refinement
t1 = time.time()
Ta, La = optimize_affine(T0, lambda T: a.w_ncc * anchor.loss_of(T) + a.w_ves * vessel_auto_loss(T), a.iters)
rows.append(evaluate(Ta, "A: auto vessels + anchor")); np.save(a.out / "T_auto_vessel.npy", Ta)
Tav, Lav = optimize_affine(T0, lambda T: vessel_auto_loss(T), a.iters)
rows.append(evaluate(Tav, "A2: auto vessels only")); np.save(a.out / "T_auto_vessel_only.npy", Tav)
say(f"[A] {time.time()-t1:.0f}s")
# B1. oracle affine from T0
Tb, Lb = optimize_affine(T0, lambda T: oracle_loss(T), a.iters, ls_clamp=0.25, sh_clamp=0.25)
rows.append(evaluate(Tb, "B1: ORACLE affine (manual)")); np.save(a.out / "T_oracle_affine.npy", Tb)
# B2. oracle nonrigid on top of B1
gshape = tuple(int(x) for x in a.grid.split(","))
disp = GridDisp(oct150.shape, A_oct, gshape)
opt = torch.optim.Adam([disp.g], lr=0.05)
Tb_t = to_t(Tb)
for it in range(a.iters):
    opt.zero_grad(); L = oracle_loss(Tb_t, disp) + 20.0 * disp.smooth(); L.backward(); opt.step()
with torch.no_grad():
    mag = disp(anchor.pts_o).norm(dim=1)
    say(f"B2 nonrigid oracle: displacement mean {mag.mean().item():.3f} mm, p95 {torch.quantile(mag, 0.95).item():.3f} mm, max {mag.max().item():.3f} mm")
    # evaluate nonrigid: vessels (MRI manual -> OCT) need inverse; approximate by evaluating OCT vessel pts -> manual MRI EDT and Dice with displaced points
    pm = apply_affine(Tb_t, anchor.pts_o) + disp(anchor.pts_o)
    wm_m = sample_at_world(LAB_WM, A_REG, pm)[0]; gm_m = sample_at_world(LAB_GM, A_REG, pm)[0]
    w = anchor.w; wm_o = (anchor.fo[0] > 0.5).float(); gm_o = (anchor.fo[1] > 0.5).float()
    dice_wm = float(2 * (w * wm_o * (wm_m > 0.5)).sum() / ((w * wm_o).sum() + (w * (wm_m > 0.5)).sum() + 1e-6))
    dice_gm = float(2 * (w * gm_o * (gm_m > 0.5)).sum() / ((w * gm_o).sum() + (w * (gm_m > 0.5)).sum() + 1e-6))
    p_mv = apply_affine(Tb_t, oct_ves_pts) + disp(oct_ves_pts)
    d_ov = sample_at_world(EDT_MAN, A_REG, p_mv, padding=5.0)[0] * 1000
    # OCT->manual-MRI vessel distances for the affine variants too (for a like-for-like comparison)
    def ov_stats(T):
        d = sample_at_world(EDT_MAN, A_REG, apply_affine(to_t(T), oct_ves_pts), padding=5.0)[0] * 1000
        return float(d.median()), float((d <= 150).float().mean())
    say(f"B2 nonrigid oracle: Dice WM {dice_wm:.3f} GM {dice_gm:.3f}; OCT-vessel -> manual-MRI-vessel median {d_ov.median().item():.0f} um (T0 {ov_stats(T0)[0]:.0f}, A {ov_stats(Ta)[0]:.0f}, B1 {ov_stats(Tb)[0]:.0f})")
    rows.append({"name": "B2: ORACLE nonrigid (manual)", "dice_WM": dice_wm, "dice_GM": dice_gm, "oct2manualves_median_um": float(d_ov.median()),
                 "disp_mean_mm": float(mag.mean()), "disp_p95_mm": float(torch.quantile(mag, 0.95)), "disp_max_mm": float(mag.max())})
    for r, T in ((rows[0], T0), (rows[1], Ta), (rows[2], Tav), (rows[3], Tb)):
        r["oct2manualves_median_um"], r["oct2manualves_f150"] = ov_stats(T)
np.save(a.out / "oracle_disp_grid.npy", disp.g.detach().cpu().numpy())
json.dump({"T0": str(a.T0), "rows": rows, "auto_vessel_candidates": int(cand.sum()), "vessel_q": a.vessel_q, "cap_mm": a.cap_mm}, open(a.out / "refine_validate.json", "w"), indent=1, default=float)
say("saved", a.out / "refine_validate.json")

#!/usr/bin/env python3
"""Label-free sub-mm refinement with an extra VASCULAR channel (real data only, no annotations):

    MRI channel  : dark-blob map of the 0.15 mm MRI (local-median darkness, or Frangi dark-tube vesselness), inside tissue
    OCT channel  : automatic vessel density of the 12 um OCT (Frangi segmentation, ves_seg) pooled to 0.15 mm
    similarity   : masked NCC over [WM, GM, vessel] channels (channel weights), affine, from T0

Motivation (refine_validate / oracle_cv / scale_probe): the tissue-class geometry of this block is blind to the scale
along the OCT depth axis (and to a ~1 mm shift along the sulcus); the manual vessels see it (held-out f150 0.25 -> 0.42),
and the authors' affine confirms a ~17% depth stretch.  Nearest-vessel distance with automatic candidates failed
(dense-vs-dense is ambiguous); a correlation of vessel *density* maps is the statistically proper alternative.
Scores every variant with the manual-vessel metric (held-out from the method), manual-label Dice, and the distance to
the oracle affine.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, avg_pool_iso, grid_points
from octreg.refine import Refiner
from octreg.features import build_features, frangi_dark_vesselness
from octreg.evaluate import transform_diff, gm_wm_overlap

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T0", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--Tref", type=Path, default=None, help="oracle affine (for distance reporting only)")
ap.add_argument("--iters", type=int, default=300); ap.add_argument("--reg", type=float, default=0.5); ap.add_argument("--clamp", type=float, default=0.3)
ap.add_argument("--weights", type=str, default="0.5,1,2,4", help="vessel-channel weights to try (relative to 1,1 for WM,GM)")
ap.add_argument("--mri-vessel", type=str, default="dark,frangi", help="MRI vessel channel variants")
ap.add_argument("--oct-vessel", type=str, default="density,frangi", help="OCT vessel channel variants (density from ves_seg, or Frangi on OCT@0.15)")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
def say(*x): print(*x, flush=True)

A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); tissue = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32); T0 = np.load(a.T0)
sh = np.array(oct150.shape); c_o = (A_oct @ np.r_[(sh - 1) / 2.0, 1.0])[:3]; centre = (T0 @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]; A_REG = to_t(A_reg)
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); tis_reg = np.asarray(tissue[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
P_WM, P_GM = to_t(load_region("wm")), to_t(load_region("gm"))
_pred = oct_prob.argmax(0); wm_bright = bool(oct150[(_pred == 1) & oct_mask].mean() > oct150[(_pred >= 2) & oct_mask].mean())
FO2 = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=wm_bright)          # [2,D,H,W]
MO = to_t(oct_mask)[None].float()
oct_class = np.where(oct_mask, np.where(FO2[0].cpu().numpy() > 0.5, 1, 2), 0)

# ---- MRI vessel channels (annotation-free)
loc = ndimage.median_filter(mri_reg, size=9)
dark = np.clip(loc - mri_reg, 0, None) * tis_reg; dark = dark / max(np.percentile(dark[tis_reg], 99.5), 1e-6); dark = np.clip(dark, 0, 1).astype(np.float32)
fr = frangi_dark_vesselness(to_t(mri_reg)[None])[0] * to_t(tis_reg.astype(np.float32)); fr = (fr / fr.flatten().kthvalue(int(0.995 * fr.numel())).values.clamp(min=1e-6)).clamp(0, 1)
MRI_V = {"dark": to_t(dark), "frangi": fr}
# ---- OCT vessel channels (annotation-free): density of the automatic 12 um segmentation pooled to ~0.15 mm, sampled on the oct150 grid
d0um = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
vb = to_t((d0um <= 0).astype(np.float32))[None]                       # 48 um binary vessels
dens48, A_d = avg_pool_iso(vb, A0, 3)                                  # ~0.144 mm density
pts = grid_points(to_t(A_oct), oct150.shape).reshape(-1, 3)
dens = sample_at_world(dens48, to_t(A_d), pts)[0].reshape(oct150.shape)
OM = to_t(oct_mask, dtype=torch.bool)
dens = (dens / dens[OM].flatten().kthvalue(int(0.995 * int(oct_mask.sum()))).values.clamp(min=1e-6)).clamp(0, 1) * to_t(oct_mask.astype(np.float32))
fo_fr = frangi_dark_vesselness(to_t(oct150)[None])[0] * to_t(oct_mask.astype(np.float32)); fo_fr = (fo_fr / fo_fr[OM].flatten().kthvalue(int(0.995 * int(oct_mask.sum()))).values.clamp(min=1e-6)).clamp(0, 1)
OCT_V = {"density": dens, "frangi": fo_fr}
say(f"OCT vessel density: mean in tissue {float(dens[OM].mean()):.3f}; MRI dark p50 {float(np.percentile(dark[tis_reg],50)):.3f}")

# ---- evaluation (manual annotations, held out from the method)
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); man = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
man_w = to_t((A_reg @ np.c_[np.argwhere(man), np.ones(int(man.sum()))].T).T[:, :3])
EDT_O = to_t(d0um / 1000.0)[None]; A_O48 = to_t(A0); oct_shape48 = to_t(np.array(d0um.shape) - 1)
R = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0)
Tref = np.load(a.Tref) if a.Tref else None
def vessel_score(T):
    with torch.no_grad():
        p_o = apply_affine(torch.linalg.inv(to_t(T)), man_w); v = apply_affine(torch.linalg.inv(A_O48), p_o)
        inside = ((v >= 0) & (v <= oct_shape48)).all(1); d = sample_at_world(EDT_O, A_O48, p_o)[0][inside] * 1000
        return float(d.median()), float((d <= 150).float().mean())
def report(name, T, extra=""):
    vm, vf = vessel_score(T); gw = gm_wm_overlap(T, oct_class, oct_mask, A_oct, np.asarray(labels4), A_mri)
    st = [round(float(np.linalg.norm(T[:3, :3] @ R[:, q])), 3) for q in range(3)]
    d0_ = transform_diff(T, T0, c_o)["corner_mean_mm"]; dr = transform_diff(T, Tref, c_o)["corner_mean_mm"] if Tref is not None else float("nan")
    say(f"{name:44s} vessels med {vm:4.0f} f150 {vf:.2f} | Dice WM {gw['dice_WM']:.3f} GM {gw['dice_GM']:.3f} | stretch i,j,k {st} | corners vs T0 {d0_:.2f} vs oracle {dr:.2f} mm {extra}")
    return {"name": name, "vessel_median_um": vm, "vessel_f150": vf, "dice_WM": gw["dice_WM"], "dice_GM": gw["dice_GM"], "stretch_ijk": st, "corner_vs_T0": d0_, "corner_vs_oracle": dr, "T": T.tolist()}

rows = [report("T0", T0)]
if Tref is not None: rows.append(report("oracle affine (manual, reference)", Tref))
weights = [float(x) for x in a.weights.split(",")]
t0 = time.time()
for mv in a.mri_vessel.split(","):
    for ov in a.oct_vessel.split(","):
        FM = torch.stack([P_WM, P_GM, MRI_V[mv]], 0); FO = torch.cat([FO2, OCT_V[ov][None]], 0)
        for w in weights:
            ref = Refiner(FM, A_reg, FO, A_oct, MO, chan_w=[1.0, 1.0, w])
            # coarse-to-fine: 0.3 mm affine then 0.15 mm affine, both with the vessel channel
            f2 = {}
            a1, A1 = avg_pool_iso(FM, A_reg, 2); a2, A2 = avg_pool_iso(FO, A_oct, 2); a3, _ = avg_pool_iso(MO, A_oct, 2)
            ref2 = Refiner(a1, A1, a2, A2, a3, chan_w=[1.0, 1.0, w])
            T, l2 = ref2.refine(T0, dof="affine", iters=a.iters, ls_clamp=a.clamp, sh_clamp=a.clamp, reg=a.reg)
            T, l1 = ref.refine(T, dof="affine", iters=a.iters, ls_clamp=a.clamp, sh_clamp=a.clamp, reg=a.reg)
            ch = ref.ncc_channels(T)
            rows.append(report(f"mri:{mv} oct:{ov} w={w}", T, extra=f"| ncc/ch {np.round(ch,3).tolist()} ({time.time()-t0:.0f}s)"))
        # vessel-only refinement (weights 0,0,1) as a control
        ref = Refiner(FM, A_reg, FO, A_oct, MO, chan_w=[0.0, 0.0, 1.0])
        T, l = ref.refine(T0, dof="affine", iters=a.iters, ls_clamp=a.clamp, sh_clamp=a.clamp, reg=a.reg)
        rows.append(report(f"mri:{mv} oct:{ov} VESSEL ONLY", T, extra=f"| ncc/ch {np.round(ref.ncc_channels(T),3).tolist()}"))
# control: same schedule without the vessel channel (is any gain just from the schedule / weaker prior?)
ref = Refiner(torch.stack([P_WM, P_GM], 0), A_reg, FO2, A_oct, MO)
a1, A1 = avg_pool_iso(torch.stack([P_WM, P_GM], 0), A_reg, 2); a2, A2 = avg_pool_iso(FO2, A_oct, 2); a3, _ = avg_pool_iso(MO, A_oct, 2)
T, _ = Refiner(a1, A1, a2, A2, a3).refine(T0, dof="affine", iters=a.iters, ls_clamp=a.clamp, sh_clamp=a.clamp, reg=a.reg)
T, _ = ref.refine(T, dof="affine", iters=a.iters, ls_clamp=a.clamp, sh_clamp=a.clamp, reg=a.reg)
rows.append(report(f"control: WM/GM only, same schedule (reg {a.reg}, clamp {a.clamp})", T))
json.dump({"T0": str(a.T0), "reg": a.reg, "clamp": a.clamp, "rows": rows}, open(a.out / "vessel_channel_refine.json", "w"), indent=1, default=float)
say("saved", a.out / "vessel_channel_refine.json")

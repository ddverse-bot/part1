#!/usr/bin/env python3
"""Which OCT vessel scale should the label-free vascular channel use?  Variants of our own Frangi segmentation
(24 um level with small/large sigmas, 48 um level, unions), scored like test_own_vessels.py."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import to_t, apply_affine, sample_at_world, world_bbox_to_voxel, avg_pool_iso
from octreg.features import build_features
from octreg.evaluate import transform_diff, gm_wm_overlap
from octreg.vascular import mri_dark_channel, mri_frangi_channel, oct_vessel_mask, density_on_grid, vascular_refine, frangi_chunked

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--parser", type=Path, required=True)
ap.add_argument("--T0", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--Tref", type=Path, default=None)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
def say(*x): print(*x, flush=True)
A_mri = np.load(a.work / "mri_affine.npy"); mri = np.load(a.work / "mri.npy", mmap_mode="r")
labels4 = np.load(a.work / "labels4.npy", mmap_mode="r"); tissue = np.load(a.work / "mri_tissue.npy", mmap_mode="r")
oct150 = np.load(a.work / "oct150.npy"); oct_mask = np.load(a.work / "oct150_mask.npy"); A_oct = np.load(a.work / "oct150_affine.npy")
oct_prob = np.load(a.parser / "oct150_prob.npy").astype(np.float32); T0 = np.load(a.T0)
sh = np.array(oct150.shape); c_o = (A_oct @ np.r_[(sh - 1) / 2.0, 1.0])[:3]; centre = (T0 @ np.r_[c_o, 1.0])[:3]
lo, hi = world_bbox_to_voxel(A_mri, mri.shape, centre - 22, centre + 22)
A_reg = A_mri.copy(); A_reg[:3, 3] = (A_mri @ np.r_[lo, 1.0])[:3]
mri_reg = np.asarray(mri[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); tis_reg = np.asarray(tissue[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
def load_region(name):
    arr = np.load(a.parser / f"mri_prob_{name}.npy", mmap_mode="r"); return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
FM2 = torch.stack([to_t(load_region("wm")), to_t(load_region("gm"))], 0)
_pred = oct_prob.argmax(0); wm_bright = bool(oct150[(_pred == 1) & oct_mask].mean() > oct150[(_pred >= 2) & oct_mask].mean())
FO2 = build_features("otsu", to_t(oct150)[None], to_t(oct_mask)[None].float(), wm_bright=wm_bright); MO = to_t(oct_mask)[None].float()
oct_class = np.where(oct_mask, np.where(FO2[0].cpu().numpy() > 0.5, 1, 2), 0)
MRI_V = {"dark": mri_dark_channel(mri_reg, tis_reg), "frangi": mri_frangi_channel(mri_reg, tis_reg)}
mri_ves = np.load(a.work / "mri_vessels.npy", mmap_mode="r"); man = np.asarray(mri_ves[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(bool)
man_w = to_t((A_reg @ np.c_[np.argwhere(man), np.ones(int(man.sum()))].T).T[:, :3])
d0um = np.load(a.work / "oct_ves_dist48_z0.npy"); A0 = np.load(a.work / "oct_ves_dist48_z0_affine.npy")
EDT_O = to_t(d0um / 1000.0)[None]; A_O48 = to_t(A0); oct_shape48 = to_t(np.array(d0um.shape) - 1)
R = A_oct[:3, :3] / np.linalg.norm(A_oct[:3, :3], axis=0); Tref = np.load(a.Tref) if a.Tref else None
def vessel_score(T):
    with torch.no_grad():
        p_o = apply_affine(torch.linalg.inv(to_t(T)), man_w); v = apply_affine(torch.linalg.inv(A_O48), p_o)
        inside = ((v >= 0) & (v <= oct_shape48)).all(1); d = sample_at_world(EDT_O, A_O48, p_o)[0][inside] * 1000
        return float(d.median()), float((d <= 150).float().mean())
rows = []
def report(name, T, extra=""):
    vm, vf = vessel_score(T); gw = gm_wm_overlap(T, oct_class, oct_mask, A_oct, np.asarray(labels4), A_mri)
    st = [round(float(np.linalg.norm(T[:3, :3] @ R[:, q])), 3) for q in range(3)]
    d0_ = transform_diff(T, T0, c_o)["corner_mean_mm"]; dr = transform_diff(T, Tref, c_o)["corner_mean_mm"] if Tref is not None else float("nan")
    say(f"{name:46s} vessels med {vm:4.0f} f150 {vf:.2f} | Dice WM {gw['dice_WM']:.3f} GM {gw['dice_GM']:.3f} | stretch {st} | vs T0 {d0_:.2f} vs oracle {dr:.2f} mm {extra}")
    rows.append({"name": name, "vessel_median_um": vm, "vessel_f150": vf, "dice_WM": gw["dice_WM"], "dice_GM": gw["dice_GM"], "stretch_ijk": st, "corner_vs_T0": d0_, "corner_vs_oracle": dr})

oct24 = np.load(a.work / "oct24.npy").astype(np.float32); m24 = np.load(a.work / "oct24_mask.npy"); A24 = np.load(a.work / "oct24_affine.npy")
x24 = to_t(oct24); t24 = to_t(m24, dtype=torch.bool)
x48, A48 = avg_pool_iso(x24[None], A24, 2); x48 = x48[0]; t48 = F.avg_pool3d(t24.float()[None, None], 2)[0, 0] > 0.5
t0 = time.time()
V = {}
V["24um s1-2.2"] = frangi_chunked(x24, (1.0, 1.5, 2.2))
V["24um s2-4.4"] = frangi_chunked(x24, (2.0, 3.0, 4.4))
V["48um s1-2.2"] = frangi_chunked(x48, (1.0, 1.5, 2.2))
V["48um s2-4.4"] = frangi_chunked(x48, (2.0, 3.0, 4.4))
say(f"vesselness maps ({time.time()-t0:.0f}s)")
def mask_top(v, t, q):
    thr = v[t].flatten().kthvalue(int(q / 100.0 * int(t.sum()))).values; return (v > thr) & t
DENS = {}
for k, v in V.items():
    lvl = 24 if k.startswith("24") else 48
    for q in (98.5, 99.0, 99.5):
        m = mask_top(v, t24 if lvl == 24 else t48, q)
        DENS[f"{k} q{q}"] = density_on_grid(m, A24 if lvl == 24 else A48, A_oct, oct150.shape, oct_mask, pool=6 if lvl == 24 else 3)
# multi-scale union at 24 um: small OR large sigmas
m_u = mask_top(V["24um s1-2.2"], t24, 99.0) | mask_top(V["24um s2-4.4"], t24, 99.0)
DENS["24um union(s1-2.2,s2-4.4) q99"] = density_on_grid(m_u, A24, A_oct, oct150.shape, oct_mask, pool=6)
m_u2 = mask_top(torch.maximum(V["24um s1-2.2"], V["24um s2-4.4"]), t24, 99.0)
DENS["24um max(s1-2.2,s2-4.4) q99"] = density_on_grid(m_u2, A24, A_oct, oct150.shape, oct_mask, pool=6)
# reference: DANDI ves_seg density
DENS["ves_seg@12um"] = density_on_grid(to_t((d0um <= 0).astype(np.float32)) > 0.5, A0, A_oct, oct150.shape, oct_mask, pool=3)
report("T0", T0)
if Tref is not None: report("oracle affine (manual, reference)", Tref)
for k, d in DENS.items():
    for mv in ("dark",):
        T, info = vascular_refine(T0, FM2, A_reg, FO2, A_oct, MO, MRI_V[mv], d, w=1.0, reg=0.5, clamp=0.3)
        report(f"{k} | mri {mv}", T, extra=f"| ncc/ch {np.round(info['ncc_channels'],3).tolist()}")
json.dump({"rows": rows}, open(a.out / "test_vessel_scales.json", "w"), indent=1, default=float)
say("saved")
